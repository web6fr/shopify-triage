"""Single-node LangGraph graph that routes a Shopify merchant ticket to billing or orders."""

import csv
import sys
from typing import Literal, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from shopify_triage.llm import get_llm

TRIAGE_PROMPT = """You triage support tickets written by Shopify merchants.

Choose exactly one agent:
- billing: money owed to or by the merchant — Shopify bills and invoices, app or subscription charges, refunds of those charges, Shopify Payments payouts, fees and taxes on the merchant's account.
- orders: the merchant's store operations — customer orders, fulfillment, shipping rates and labels, tracking, returns and order edits.

Return your decision with three fields:
- agent: billing or orders
- reason: one sentence that points to what the merchant wrote
- confidence: a number from 0 to 1 for how sure you are of the agent"""


class RouteDecision(BaseModel):
    """Which specialist agent should handle the ticket, and why."""

    agent: Literal["billing", "orders"] = Field(description="The specialist agent that should handle the ticket.")
    reason: str = Field(min_length=1, description="One sentence that points to what the merchant wrote.")
    confidence: float = Field(ge=0, le=1, description="How sure the model is of the agent, from 0 to 1.")


class TriageState(TypedDict, total=False):
    message_body: str
    decision: RouteDecision


class InvalidRouteError(ValueError):
    """The model did not return a valid route decision (agent must be billing or orders)."""


def triage(state: TriageState) -> TriageState:
    llm = get_llm(temperature=0).with_structured_output(RouteDecision, method="function_calling")
    try:
        decision = llm.invoke([SystemMessage(TRIAGE_PROMPT), HumanMessage(state["message_body"])])
    except (ValidationError, OutputParserException) as exc:
        raise InvalidRouteError(f"Model returned an invalid route decision: {exc}") from exc
    if decision is None:
        raise InvalidRouteError("Model returned no route decision.")
    return {"decision": decision}


builder = StateGraph(TriageState)
builder.add_node("triage", triage)
builder.add_edge(START, "triage")
builder.add_edge("triage", END)
graph = builder.compile()


if __name__ == "__main__":
    # Usage: python triage.py [ticket_id ...]  (defaults to every ticket in the test set)
    wanted = set(sys.argv[1:])
    with open("shopify_merchant_tickets_test_set.csv", newline="", encoding="utf-8") as f:
        tickets = [t for t in csv.DictReader(f) if not wanted or t["ticket_id"] in wanted]
    for ticket in tickets:
        decision = graph.invoke({"message_body": ticket["message_body"]})["decision"]
        print(f"{ticket['ticket_id']}  agent={decision.agent}  expected={ticket['expected_route']}  confidence={decision.confidence:.2f}")
        print(f"  Reason: {decision.reason}")

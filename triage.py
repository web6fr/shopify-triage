"""Single-node LangGraph graph that routes a Shopify merchant ticket to billing or orders."""

import csv
import re
import sys
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from shopify_triage.llm import get_llm

VALID_ROUTES = {"billing", "orders"}

TRIAGE_PROMPT = """You triage support tickets written by Shopify merchants.

Choose exactly one route:
- billing: money owed to or by the merchant — Shopify bills and invoices, app or subscription charges, refunds of those charges, Shopify Payments payouts, fees and taxes on the merchant's account.
- orders: the merchant's store operations — customer orders, fulfillment, shipping rates and labels, tracking, returns and order edits.

Answer with exactly these two labelled lines and nothing else:
Route: <billing or orders>
Reason: <one sentence that points to what the merchant wrote>"""

ROUTE_LINE = re.compile(r"^[\s*_#>-]*route[\s*_]*:[\s*_]*(.+)$", re.IGNORECASE | re.MULTILINE)
REASON_LINE = re.compile(r"^[\s*_#>-]*reason[\s*_]*:[\s*_]*(.+)$", re.IGNORECASE | re.MULTILINE)


class TriageState(TypedDict, total=False):
    message_body: str
    route: str
    reason: str


class InvalidRouteError(ValueError):
    """The model replied with a route other than billing or orders."""


def parse_reply(reply: str) -> tuple[str, str]:
    route_match = ROUTE_LINE.search(reply)
    if not route_match:
        raise InvalidRouteError(f"Model reply has no 'Route:' line:\n{reply}")
    route = route_match.group(1).strip().strip("*_`.\"' ").lower()
    if route not in VALID_ROUTES:
        raise InvalidRouteError(
            f"Model returned route {route_match.group(1).strip()!r}; expected 'billing' or 'orders'.\nFull reply:\n{reply}"
        )
    reason_match = REASON_LINE.search(reply)
    reason = reason_match.group(1).strip().strip("*_`") if reason_match else ""
    return route, reason


def triage(state: TriageState) -> TriageState:
    reply = get_llm(temperature=0).invoke(
        [SystemMessage(TRIAGE_PROMPT), HumanMessage(state["message_body"])]
    )
    route, reason = parse_reply(reply.content)
    return {"route": route, "reason": reason}


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
        result = graph.invoke({"message_body": ticket["message_body"]})
        print(f"{ticket['ticket_id']}  route={result['route']}  expected={ticket['expected_route']}")
        print(f"  Reason: {result['reason']}")

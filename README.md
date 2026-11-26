# shopify-triage

LangGraph agent for triaging Shopify merchant support tickets, using models served through OpenRouter.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set OPENROUTER_API_KEY (OPENROUTER_MODEL defaults to a free model)
```

## Data

`shopify_merchant_tickets_test_set.csv` — 23 labelled test tickets with the expected route, tool calls, fix, and escalation flag for each.

## Triage

`triage.py` builds a LangGraph graph with a single node, `triage`. It sends a ticket's `message_body` to the model named in `OPENROUTER_MODEL` (key from `OPENROUTER_API_KEY`, both read from `.env`) and returns a Pydantic `RouteDecision`:

```python
class RouteDecision(BaseModel):
    agent: Literal["billing", "orders"]
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
```

The model fills the decision through structured output (`with_structured_output(RouteDecision, method="function_calling")`). If it returns an agent other than `billing` or `orders`, an empty reason, a confidence outside 0–1, or no decision at all, the graph stops with an `InvalidRouteError`.

```bash
python triage.py SUP-48213          # one or more ticket ids
python triage.py                    # every ticket in the test set
```

### Route decision on Maria's ticket (SUP-48213, Maria Sousa)

Model: `nvidia/nemotron-3.5-super-vl:free`

```
$ .venv/bin/python triage.py SUP-48213
SUP-48213  agent=orders  expected=orders  confidence=0.97
  Reason: The merchant reports that shipping rates are incorrectly applied to customer orders, with domestic Portugal and international European rates swapped across about 40 orders since Black Friday.
```

**Verify:** all three fields are filled. `agent` is `orders`, which matches expected_route `orders` ✅; `reason` is one sentence pointing to her shipping-rate complaint ✅; `confidence` is `0.97`, within 0–1 ✅.

## Mock Shopify tools

`shopify_triage/shopify_tools.py` is a mock Shopify Admin API: plain Python functions over an in-memory `FIXTURES` store of four shops. Every tool call is appended to `CALL_LOG`, and `reset_store()` restores the fixtures and clears the log between tickets.

It has 16 tools: **8 read** and **8 write**. A tool counts as a write if it changes the store. This was checked by running each tool on a freshly reset store and comparing `FIXTURES` before and after the call.

| Tool | Read or write | Agent |
|---|---|---|
| `get_shop` | read | billing and orders |
| `get_order` | read | orders |
| `list_orders` | read | orders |
| `list_order_transactions` | read | orders |
| `calculate_refund` | read | orders |
| `create_refund` | write | orders |
| `get_delivery_profile` | read | orders |
| `update_shipping_rate` | write | orders |
| `recalculate_order_shipping` | write | orders |
| `update_order_shipping_address` | write | orders |
| `cancel_order` | write | orders |
| `add_order_note` | write | orders |
| `get_app_subscriptions` | read | billing |
| `list_billing_charges` | read | billing |
| `cancel_app_subscription` | write | billing |
| `issue_billing_credit` | write | billing |

The orders agent gets 6 read and 6 write tools; the billing agent gets 3 read and 2 write tools (`get_shop` is shared). `create_refund` refunds the buyer on an order, while `issue_billing_credit` credits the merchant's Shopify bill, so they belong to different agents. Two names can mislead: `calculate_refund` only previews a refund and is a read, while `recalculate_order_shipping` saves a new shipping price onto an existing order and is a write.

`reset_store()` also changes the store, but it is used by the graph between tickets and is not given to either agent.

## Stage 1: two-line text reply

The sections below record the first version of the triage node, which asked the model for two labelled lines (`Route:` and `Reason:`) and read the route from the text. The prompt's route definitions are unchanged in the current version; only the output format moved to `RouteDecision`.

### Key part of the triage prompt (stage 1)

```
Choose exactly one route:
- billing: money owed to or by the merchant — Shopify bills and invoices, app or subscription charges, refunds of those charges, Shopify Payments payouts, fees and taxes on the merchant's account.
- orders: the merchant's store operations — customer orders, fulfillment, shipping rates and labels, tracking, returns and order edits.

Answer with exactly these two labelled lines and nothing else:
Route: <billing or orders>
Reason: <one sentence that points to what the merchant wrote>
```

The full prompt is `TRIAGE_PROMPT` in `triage.py`.

### Run on Maria's ticket (SUP-48213, Maria Sousa), stage 1

Model: `nvidia/nemotron-3.5-super-vl:free`

```
$ .venv/bin/python triage.py SUP-48213
SUP-48213  route=orders  expected=orders
  Reason: The merchant reports that shipping rates are incorrect for customer orders, with buyers in Europe charged the domestic Portugal rate and buyers in Portugal charged international rates.
```

**Verify:** route `orders` == expected_route `orders` → ✅ match

**First run:** the route did not disagree. The first two attempts, with `google/gemma-4-31b-it:free`, returned no route at all: both failed with `OpenAIRateLimitError: Error code: 429 - Provider returned error` because the free model's provider was refusing requests. After switching `OPENROUTER_MODEL` to `nvidia/nemotron-3.5-super-vl:free`, the first run that returned a route is the one shown above, and it matched. No re-run or mismatch reasoning is needed.

### Contrasting ticket (SUP-48204, Inês Carvalho, expected `billing`)

Model: `nvidia/nemotron-3.5-super-vl:free`

```
$ .venv/bin/python triage.py SUP-48204
SUP-48204  route=billing  expected=billing
  Reason: The merchant is asking about a Shopify Payments payout marked as in transit that has not arrived in their bank account.
```

**Verify:** route `billing` == expected_route `billing` → ✅ match

### Side by side

| | Maria's ticket | Contrasting ticket |
|---|---|---|
| Ticket | SUP-48213, Maria Sousa | SUP-48204, Inês Carvalho |
| Subject | shipping is broken since the weekend | payout? |
| Expected route | orders | billing |
| Model's route | **orders** ✅ | **billing** ✅ |
| Reason | The merchant reports that shipping rates are incorrect for customer orders, with buyers in Europe charged the domestic Portugal rate and buyers in Portugal charged international rates. | The merchant is asking about a Shopify Payments payout marked as in transit that has not arrived in their bank account. |

### Model choice

**Model in use:** `nvidia/nemotron-3.5-super-vl:free`, set by `OPENROUTER_MODEL` in `.env`.

**Why I chose it:** I switched to `nvidia/nemotron-3.5-super-vl:free` because the provider behind `google/gemma-4-31b-it:free` kept refusing requests with 429 rate limits, and Nemotron is on OpenRouter's free list with tool-calling support, which the agent will need for the test set's `expected_tool_calls`.

### A phrase in Maria's message that could pull toward billing

> "one customer from Braga paid €14,90 postage for a small mug that costs €4,50 to send inside portugal"

The phrase is about a customer paying too much, with euro amounts, so a model could read it as a charge or refund question and send it to billing. The actual problem is the store's shipping rates, which belongs to orders.

**How I changed the prompt to handle it:** I didn't change it after the first run: the prompt I first wrote already defines `orders` to include "shipping rates and labels" and limits `billing` to the merchant's own Shopify bills, app charges and payouts, so a customer overpaying for postage reads as a shipping-rate problem, and Maria's ticket routed to `orders` as expected.

### Reviewer notes

**What I checked or changed in the code:** I checked the route for Maria Sousa's ticket SUP-48213: the model returned `orders`, and her `expected_route` in the test set is `orders`, so they match. The first two runs never produced a wrong route; they produced no route at all, because `google/gemma-4-31b-it:free` failed both times with a 429 rate limit from its provider before answering. The one thing I changed was `OPENROUTER_MODEL` in `.env`, from `google/gemma-4-31b-it:free` to `nvidia/nemotron-3.5-super-vl:free`; I did not change `triage.py` or the prompt.

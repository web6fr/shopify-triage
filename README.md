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

`triage.py` builds a LangGraph graph with a single node, `triage`. It sends a ticket's `message_body` to the model named in `OPENROUTER_MODEL` (key from `OPENROUTER_API_KEY`, both read from `.env`), takes the route from the model's `Route:` line, and stops with an `InvalidRouteError` if the route is anything other than `billing` or `orders`.

```bash
python triage.py SUP-48213          # one or more ticket ids
python triage.py                    # every ticket in the test set
```

### Key part of the triage prompt

```
Choose exactly one route:
- billing: money owed to or by the merchant — Shopify bills and invoices, app or subscription charges, refunds of those charges, Shopify Payments payouts, fees and taxes on the merchant's account.
- orders: the merchant's store operations — customer orders, fulfillment, shipping rates and labels, tracking, returns and order edits.

Answer with exactly these two labelled lines and nothing else:
Route: <billing or orders>
Reason: <one sentence that points to what the merchant wrote>
```

The full prompt is `TRIAGE_PROMPT` in `triage.py`.

### Run on Maria's ticket (SUP-48213, Maria Sousa)

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

**Why I chose it:** _TODO — explain why you chose this free model._

### A phrase in Maria's message that could pull toward billing

> "one customer from Braga paid €14,90 postage for a small mug that costs €4,50 to send inside portugal"

The phrase is about a customer paying too much, with euro amounts, so a model could read it as a charge or refund question and send it to billing. The actual problem is the store's shipping rates, which belongs to orders.

**How I changed the prompt to handle it:** _TODO — describe the change you made to the prompt and why._

### Reviewer notes

**What Claude checked:** the graph has one node (`__start__ → triage → __end__`); `parse_reply` accepts `**Route:** Billing.` as `billing` and raises `InvalidRouteError` for `Route: shipping` and for a reply with no `Route:` line; `.env` is git-ignored; live runs on SUP-48197, SUP-48213 and SUP-48204 each matched their expected_route.

**What I checked or changed in the code:** _TODO — describe what you checked or changed._

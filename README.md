# shopify-triage

LangGraph agent for triaging Shopify merchant support tickets, using models served through OpenRouter.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OpenRouter API key
```

## Data

`shopify_merchant_tickets_test_set.csv` — 23 labelled test tickets with the expected route, tool calls, fix, and escalation flag for each.

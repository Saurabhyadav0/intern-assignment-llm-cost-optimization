# ShopAssist cost lab

Both questions are under the point targets and still return the expected `action` and
`kb_article` on the four public tickets and on T-1091.

| Track | Baseline | Now (worst public ticket) | Target |
|---|---|---|---|
| Q1, single call | 8,931 pts (T-1001) | 1,620 pts | 3,000 |
| Q2, pipeline | over 70,000 pts | 1,536 pts | 2,500 |

How to run it:

```bash
pip install -r requirements.txt
python question1/run.py
python question2/run.py
python report.py
pytest -q
```

`report.py` prints the per-call breakdown for every ticket, which is where all the
numbers below come from.

The two cost tests used to assert that the baseline was expensive (`> 15000` and
`> 70000`), so they failed as soon as the work was done. I changed them to assert the
targets instead (`<= 3000` per ticket for Q1, `<= 2500` for Q2), which keeps the suite
green and turns them into regression guards. All 6 tests pass.

## Question 1

The baseline spent 522 input tokens and 491 output tokens on T-1001. Output costs 15
points a token and input costs 3, so 7,365 of the 8,931 points were the completion.
The prompt asked for step by step reasoning and quoted articles, and the client
answers that request with a ~480 token analysis trace plus the JSON. Almost the whole
bill was narration that no graded field needed.

Changes in `question1/agent.py`:

1. The instructions ask for the JSON only, so the reply is 36 to 46 tokens instead of
   about 490. Biggest win, and nothing was lost because the trace never carried a
   graded field.
2. The five help articles were replaced with the routing rules themselves. The
   decision is a short ordered list, first match wins, so the model does not have to
   work out precedence from article text.
3. The order snapshot is passed as compact sorted JSON instead of `indent=2`, about a
   third of the tokens, and the ticket metadata that no rule reads is dropped.
4. The `uuid4()` in the system prompt is gone. A random run id makes prefix caching
   impossible for every call.

T-1001 after the change:

```
handle_ticket input=310 cached=0 output=46 points=1620
  310 x 3  = 930
   46 x 15 = 690
```

The public set is 1,620 / 1,497 / 1,458 / 1,419, so the worst ticket uses 54% of the
target.

## Question 2

The baseline sent a 120k token slice of the expanded help center to triage and an 80k
slice to audit, asked three of the four nodes for reasoning prose, and passed the
triage output to compose twice. Building the expanded blob takes about three minutes
on its own, which was a hint that it did not belong in the request path.

The pipeline still has four nodes, but only one of them needs the model:

| Node | Before | Now |
|---|---|---|
| triage | 120k token slice, restate the ticket step by step | keyword cues plus order facts over five intents, no model call |
| retrieve | model call to print article text | `load_article()` on the routed ids, no model call |
| compose | facts, triage twice, article blob, full reasoning | one call: intent, thread, compact facts, two short articles |
| audit | 80k token slice, double-check and restate | local consistency check, local recovery |

Triage and retrieval are lookups over five short files. The article ids are known, so
paying 3 points a token to have the model echo them back was the most expensive and
least reliable part of the pipeline.

The audit node rejects a draft when the action is unknown, when the article cannot
support the action (`REFUND` with `digital-license-no-refund`), or when the article was
not retrieved for this ticket. It then rebuilds the pair from the order facts and the
retrieved articles.

Recovery is local rather than a retry call. I tried the retry first and measured it: a
trimmed repair prompt still cost 1,362 points, so compose plus repair was 2,898 for
T-1001, over the 2,500 target. One call already costs about 1,300, so no two call
ticket fits the budget. Because recovery is local, a rejected draft costs the same as a
clean one. I checked that path by forcing a wrong but parseable draft through compose:
T-1004, T-1003 and T-1091 all came back with the expected pair and a single `compose`
call. If there is no retrieved evidence to rebuild from, it raises instead of guessing.

T-1004, the physical replacement path:

```
compose input=258 cached=0 output=36 points=1314
  258 x 3  = 774
   36 x 15 = 540
```

T-1003, the escalation path:

```
compose input=256 cached=0 output=39 points=1353
  256 x 3  = 768
   39 x 15 = 585
```

The public set is 1,536 / 1,287 / 1,353 / 1,314, so the worst ticket uses 61% of the
target.

## Other notes

- Caching the help center instead of dropping it does not help. Cached input still
  costs 0.3 points a token, so 210k cached tokens is about 63k points per ticket, and
  it needs the same prefix on every call including the first, with a cold cache for
  each process.
- I kept `customer_reply` in the Q1 JSON even though it is about 600 of the 1,620
  points. The product sends a reply to the customer, and removing it would be cutting
  cost by cutting the feature. The target works without that.
- Neither track reaches the 1,024 token cache floor, so `cached_prefix_tokens` is 0.
  The prompts have a fixed head and a variable tail now, so they will cache once the
  rule set grows.
- Q1 still raises on an unparseable answer rather than defaulting to an action, so a
  bad generation escalates instead of refunding the wrong customer.
- No batching across tickets. Requests are independent and a shared prefix would have
  to clear 1,024 tokens before it saved anything, which would mean padding the prompt
  with text the model does not need.

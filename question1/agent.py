from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


# Ordered rules, first match wins. The model follows the list instead of comparing
# five articles itself, which keeps the prompt small and the answer stable.
POLICY = """You are ShopAssist, the Tier-1 support copilot. Pick one action and one article id.

Routing rules, first match wins:
1. product_type subscription: subscription_age_days <= 30 -> REFUND, else SEND_ARTICLE; article subscription-cooling-off.
2. license_revealed true -> SEND_ARTICLE; article digital-license-no-refund.
3. customized true and damage_reported true -> ESCALATE_TIER2; article custom-items-no-replace.
4. damage_reported true, customized false, delivered_days_ago <= 7 -> REPLACE; article damage-replacement-7-days.
5. product_type physical, opened false, delivered_days_ago <= 30 -> REFUND; article returns-within-30-days.
6. anything else -> ESCALATE_TIER2; article returns-within-30-days.

Reply with one JSON object: {"action": "...", "kb_article": "...", "customer_reply": "..."}"""


class SupportCopilot:
    """ShopAssist Tier-1 copilot.

    One call per ticket. Input costs 3 points a token and output costs 15, so the
    prompt only carries the rules plus the facts the rules read, and it asks for the
    JSON only. The original prompt asked for step by step reasoning and quoted
    articles, which turned a 1.5k point ticket into a 9k point ticket.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])

        # Compact snapshot: same fields, roughly a third of the tokens of indent=2.
        snapshot = json.dumps(order, separators=(",", ":"), sort_keys=True)
        user = "\n\n".join(
            [
                f"THREAD:\n{format_thread(ticket)}",
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ]
        )
        return [ChatMessage("system", POLICY), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        completion = self.llm.chat("handle_ticket", self.build_messages(ticket_id))
        return parse_resolution(completion)

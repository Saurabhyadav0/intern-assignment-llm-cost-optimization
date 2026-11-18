from __future__ import annotations

import json
from dataclasses import dataclass, field

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution


@dataclass
class TicketResolution:
    action: str
    kb_article: str


@dataclass
class TicketRoute:
    """Triage result: what the customer wants and which articles to pull."""

    intent: str
    article_ids: list[str] = field(default_factory=list)


# Cues per intent. There are only five articles and five intents, so this is a
# closed problem and does not need a model call.
INTENT_CUES: dict[str, tuple[str, ...]] = {
    "cancellation": ("cancel", "subscription", "plan", "annual", "billing period"),
    "digital_licence": ("ebook", "licence", "license", "download", "digital", "key"),
    "damage": ("cracked", "damaged", "damage", "broken", "showed up", "arrived"),
    "replacement": ("replacement", "replace", "send another", "swap"),
    "refund": ("refund", "money back", "return", "unopened", "never opened"),
}

INTENT_ARTICLES: dict[str, tuple[str, ...]] = {
    "cancellation": ("subscription-cooling-off",),
    "digital_licence": ("digital-license-no-refund",),
    "damage": ("damage-replacement-7-days", "custom-items-no-replace"),
    "replacement": ("damage-replacement-7-days", "custom-items-no-replace"),
    "refund": ("returns-within-30-days", "digital-license-no-refund"),
}

# Pairs the audit node accepts.
ACTION_ARTICLES: dict[str, tuple[str, ...]] = {
    "REFUND": ("returns-within-30-days", "subscription-cooling-off"),
    "REPLACE": ("damage-replacement-7-days",),
    "SEND_ARTICLE": (
        "digital-license-no-refund",
        "subscription-cooling-off",
        "returns-within-30-days",
    ),
    "ESCALATE_TIER2": ("custom-items-no-replace", "returns-within-30-days"),
}

# Articles that only ever go with one action. The other two depend on order facts
# and are handled in SupportPipeline._recover.
_ACTION_FOR_ARTICLE: dict[str, str] = {
    "digital-license-no-refund": "SEND_ARTICLE",
    "custom-items-no-replace": "ESCALATE_TIER2",
    "damage-replacement-7-days": "REPLACE",
}

COMPOSE_POLICY = """You are ShopAssist's compose node for a Tier-1 e-commerce support desk.
Use the order facts and the help articles provided. Pick one action from
REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2 and cite the single article id that
justifies it. Reply with only:
{"action": "...", "kb_article": "...", "customer_reply": "..."}"""


class SupportPipeline:
    """Same product as Question 1, run as triage, retrieve, compose, audit.

    The expanded help center is 210k tokens of mostly filler, and the old pipeline
    sent a slice of it to triage and another to audit. Triage and retrieval are
    lookups over five short files, so they run locally; compose is the only model
    call, and audit checks the answer against the retrieved articles and rebuilds it
    from the order facts if it does not hold up.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.articles = article_ids()

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"), sort_keys=True)

        route = self._triage(thread, order)
        retrieved = self._retrieve(route)
        draft = self._compose(thread, snapshot, route, retrieved)
        final = self._audit(order, retrieved, draft)
        parsed = parse_resolution(final)
        return TicketResolution(parsed.action, parsed.kb_article)

    # node 1: triage, local
    def _triage(self, thread: str, order: dict) -> TicketRoute:
        lower = thread.lower()
        scored = [
            (sum(1 for cue in cues if cue in lower), intent)
            for intent, cues in INTENT_CUES.items()
        ]
        scored.sort(reverse=True)
        intent = scored[0][1] if scored[0][0] else "refund"

        candidates = list(INTENT_ARTICLES[intent])
        # Order facts win over the keyword guess, so the articles we retrieve always
        # belong to the branch the order actually falls into.
        if order.get("product_type") == "subscription":
            candidates.insert(0, "subscription-cooling-off")
        if order.get("license_revealed"):
            candidates.insert(0, "digital-license-no-refund")
        if order.get("customized") and order.get("damage_reported"):
            candidates.insert(0, "custom-items-no-replace")
        elif order.get("damage_reported"):
            candidates.insert(0, "damage-replacement-7-days")
        return TicketRoute(intent=intent, article_ids=list(dict.fromkeys(candidates)))

    # node 2: retrieve, local
    def _retrieve(self, route: TicketRoute) -> list[tuple[str, str]]:
        # Five short files. Loading the two routed ids is cheaper and more accurate
        # than ranking a blob that is mostly routing filler.
        return [
            (aid, load_article(aid))
            for aid in route.article_ids[:2]
            if aid in self.articles
        ]

    # node 3: compose, the only model call
    def _compose(
        self,
        thread: str,
        snapshot: str,
        route: TicketRoute,
        retrieved: list[tuple[str, str]],
    ) -> str:
        evidence = "\n".join(f"--- {aid} ---\n{text.strip()}" for aid, text in retrieved)
        user = "\n\n".join(
            [
                f"INTENT: {route.intent}",
                f"THREAD:\n{thread}",
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
                f"ARTICLES:\n{evidence}",
            ]
        )
        messages = [ChatMessage("system", COMPOSE_POLICY), ChatMessage("user", user)]
        return self.llm.chat("compose", messages)

    # node 4: audit, local
    def _audit(
        self,
        order: dict,
        retrieved: list[tuple[str, str]],
        draft: str,
    ) -> str:
        try:
            parsed = parse_resolution(draft)
        except (ValueError, KeyError):
            return self._recover(order, retrieved)

        allowed = ACTION_ARTICLES.get(parsed.action)
        if allowed is None:
            return self._recover(order, retrieved)
        if parsed.kb_article not in allowed:
            return self._recover(order, retrieved)
        if retrieved and parsed.kb_article not in {aid for aid, _ in retrieved}:
            return self._recover(order, retrieved)
        return draft

    def _recover(self, order: dict, retrieved: list[tuple[str, str]]) -> str:
        """Rebuild the pair from the order facts and the retrieved articles.

        A second model call costs about 1300 points on its own, and 1300 plus a
        compose call is over the 2500 budget even with a short retry prompt, so
        recovery runs locally. Only two articles need a fact check here.
        """
        for aid, _ in retrieved:
            if aid == "returns-within-30-days":
                refundable = (
                    order.get("product_type") == "physical"
                    and not order.get("opened")
                    and int(order.get("delivered_days_ago", 99)) <= 30
                )
                action = "REFUND" if refundable else "ESCALATE_TIER2"
            elif aid == "subscription-cooling-off":
                in_window = int(order.get("subscription_age_days", 99)) <= 30
                action = "REFUND" if in_window else "SEND_ARTICLE"
            else:
                action = _ACTION_FOR_ARTICLE.get(aid)
            if action is None:
                continue
            return json.dumps(
                {"action": action, "kb_article": aid, "customer_reply": ""}
            )
        raise ValueError(
            f"audit could not recover a resolution for {order.get('order_id')}"
        )

"""Print per-call token/points breakdowns for both questions.

    python report.py

Reproduces the numbers in SOLUTION.md from the current agents.
"""

from __future__ import annotations

from question1.agent import SupportCopilot
from question2.agent import SupportPipeline

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]
HOLDOUT = "T-1091"


def breakdown(title: str, agent, ticket_ids: list[str], target: int) -> None:
    print(f"\n== {title} (target <= {target} points/ticket)")
    for tid in ticket_ids:
        start = len(agent.session.calls)
        res = agent.handle_ticket(tid)
        calls = agent.session.calls[start:]
        points = sum(c.points() for c in calls)
        print(f"\n{tid}: {res.action} / {res.kb_article} -> {points} points")
        for c in calls:
            print(
                f"  {c.label:<12} input={c.input_tokens:>4} "
                f"cached={c.cached_prefix_tokens:>4} output={c.output_tokens:>4} "
                f"points={c.points():>5}"
            )


def main() -> None:
    breakdown("Question 1", SupportCopilot(), PUBLIC + [HOLDOUT], 3000)
    breakdown("Question 2", SupportPipeline(), PUBLIC + [HOLDOUT], 2500)


if __name__ == "__main__":
    main()

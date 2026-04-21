"""Visualization tools for understanding maps and trend charts."""

import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from state import db


def generate_understanding_map(topic: str, scores: dict[str, float], save_path: str = None) -> str:
    """
    Generate a horizontal bar chart showing understanding per sub-topic.

    Args:
        topic: Main topic name
        scores: Dict of {sub_topic: score} where score is 0.0-1.0
        save_path: Optional path to save the image

    Returns:
        Path to saved image file
    """
    if not save_path:
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "state",
            f"understanding_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        )

    sub_topics = list(scores.keys())
    values = [scores[st] * 100 for st in sub_topics]

    # Color based on score
    colors = []
    for v in values:
        if v >= 80:
            colors.append("#22c55e")  # green
        elif v >= 50:
            colors.append("#f59e0b")  # amber
        else:
            colors.append("#ef4444")  # red

    fig, ax = plt.subplots(figsize=(10, max(3, len(sub_topics) * 0.6)))
    bars = ax.barh(sub_topics, values, color=colors, height=0.6)

    # Add score labels on bars
    for bar, val in zip(bars, values):
        label = f"{val:.0f}%"
        x_pos = bar.get_width() + 1
        if bar.get_width() < 15:
            x_pos = bar.get_width() + 1
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, label,
                va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Understanding Level (%)")
    ax.set_title(f"Understanding Map: {topic}", fontsize=14, fontweight="bold", pad=15)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#22c55e", label="Strong (80%+)"),
        mpatches.Patch(facecolor="#f59e0b", label="Developing (50-80%)"),
        mpatches.Patch(facecolor="#ef4444", label="Needs Work (<50%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


def generate_trend_chart(topic: str, save_path: str = None) -> str:
    """
    Generate a trend chart from knowledge model history.

    Returns:
        Path to saved image file, or empty string if no data.
    """
    knowledge = db.get_knowledge(topic)
    if not knowledge:
        return ""

    if not save_path:
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "state",
            f"trend_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        )

    fig, ax = plt.subplots(figsize=(8, 4))

    for entry in knowledge:
        sub = entry["sub_topic"]
        score = entry["score"] * 100
        trend = entry["trend"]

        marker = "o"
        color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 50 else "#ef4444"

        trend_symbol = ""
        if trend == "improving":
            trend_symbol = " ^"
        elif trend == "declining":
            trend_symbol = " v"
        elif trend == "stable":
            trend_symbol = " ="

        ax.barh(f"{sub}{trend_symbol}", score, color=color, height=0.5)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (%)")
    ax.set_title(f"Knowledge Trend: {topic}", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


def scores_to_rich_text(topic: str, scores: dict[str, float], verdict: str, gaps: list[str]) -> str:
    """
    Generate a Rich-compatible text representation of understanding map.
    Used for terminal display.
    """
    lines = [f"\n[bold]Understanding Map: {topic}[/bold]\n"]

    for sub_topic, score in scores.items():
        pct = int(score * 100)
        filled = int(pct / 10)
        empty = 10 - filled

        if pct >= 80:
            color = "green"
            label = "strong"
        elif pct >= 50:
            color = "yellow"
            label = "developing"
        else:
            color = "red"
            label = "needs work"

        bar = f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"
        lines.append(f"  {bar}  {sub_topic:<30} {label} ({pct}%)")

    # Verdict
    verdict_color = "green" if verdict == "mastered" else "yellow" if verdict == "needs_review" else "red"
    lines.append(f"\n[bold {verdict_color}]Verdict: {verdict.upper().replace('_', ' ')}[/bold {verdict_color}]")

    # Gaps
    if gaps:
        lines.append("\n[bold]Identified gaps:[/bold]")
        for gap in gaps:
            lines.append(f"  [dim]- {gap}[/dim]")

    return "\n".join(lines)

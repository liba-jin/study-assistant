"""
StudyNet — Personal Study Agent Network

A two-agent system for adaptive personal learning:
  - Study Planner Agent: goal clarification, plan generation, adaptation
  - Test/Teacher Agent: adaptive assessment, visualization, mastery verdicts

Usage:
  python main.py                    Start interactive session
  python main.py --test <topic>     Jump directly to testing a topic
"""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.theme import Theme

from agents.orchestrator import Orchestrator
from state.db import init_db

# Rich console with custom theme
theme = Theme({
    "planner": "bold cyan",
    "tester": "bold magenta",
    "info": "dim",
    "success": "bold green",
    "warning": "bold yellow",
})
console = Console(theme=theme)


import re


def _parse_progress(text: str) -> tuple[int, int] | None:
    """Parse 'Question X of Y' from test agent response."""
    match = re.search(r"[Qq]uestion\s+(\d+)\s+of\s+(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def print_banner():
    """Print the StudyNet welcome banner."""
    banner = """
[bold cyan]StudyNet[/bold cyan] — Personal Study Agent Network

[dim]Two AI agents that help you learn anything:
  [cyan]Study Planner[/cyan] — Creates and adapts your learning roadmap
  [magenta]Test/Teacher[/magenta]  — Assesses your understanding with Socratic dialogue

Commands:
  [bold]/test <topic>[/bold]  — Test yourself on a topic directly
  [bold]/plan[/bold]          — Show current study plan
  [bold]/status[/bold]        — Show which agent is active
  [bold]/switch[/bold]        — Switch back to Study Planner
  [bold]/quit[/bold]          — Exit StudyNet
[/dim]"""
    console.print(Panel(banner, border_style="cyan"))


def get_agent_label(agent_name: str) -> str:
    """Get a colored label for an agent."""
    if agent_name == "planner":
        return "[planner]Study Planner[/planner]"
    elif agent_name == "tester":
        return "[tester]Test/Teacher[/tester]"
    return agent_name


def handle_command(command: str, orchestrator: Orchestrator) -> bool:
    """
    Handle slash commands. Returns True if a command was handled.
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/quit" or cmd == "/exit":
        console.print("\n[dim]Goodbye! Keep learning.[/dim]\n")
        sys.exit(0)

    elif cmd == "/test":
        if not arg:
            console.print("[warning]Usage: /test <topic>[/warning]")
            return True
        response, metadata = orchestrator.start_test_directly(arg)
        label = get_agent_label("tester")
        console.print(f"\n{label}: {response}\n")
        return True

    elif cmd == "/plan":
        from state.db import get_active_plan, get_plan_topics
        plan = get_active_plan()
        if not plan:
            console.print("[info]No active plan. Tell me what you want to learn![/info]")
            return True

        console.print(f"\n[bold]Current Plan:[/bold] {plan['goal']}")
        if plan["refined_goal"]:
            console.print(f"[info]Refined: {plan['refined_goal']}[/info]")
        topics = get_plan_topics(plan["id"])
        for t in topics:
            status_icon = {
                "pending": "  ",
                "in_progress": "  ",
                "assessed": "  ",
                "completed": "  ",
            }.get(t["status"], "  ")
            console.print(f"  {status_icon} {t['sequence']}. {t['topic']} [{t['status']}]")
        console.print()
        return True

    elif cmd == "/status":
        status = orchestrator.get_status()
        agent = get_agent_label(status["active_agent"])
        console.print(f"\n[info]Active agent:[/info] {agent}")
        if status["is_testing"]:
            console.print(f"[info]Testing topic:[/info] {status['tester_topic']}")
        console.print()
        return True

    elif cmd == "/switch":
        msg = orchestrator.switch_to_planner()
        label = get_agent_label("planner")
        console.print(f"\n{label}: {msg}\n")
        return True

    return False


def main():
    """Main interactive loop."""
    parser = argparse.ArgumentParser(description="StudyNet — Personal Study Agent Network")
    parser.add_argument("--test", type=str, help="Jump directly to testing a topic")
    args = parser.parse_args()

    # Initialize
    init_db()
    orchestrator = Orchestrator()

    print_banner()

    # Handle --test flag
    if args.test:
        response, metadata = orchestrator.start_test_directly(args.test)
        label = get_agent_label("tester")
        console.print(f"\n{label}: {response}\n")

    # Main loop
    while True:
        try:
            # Show which agent is active
            status = orchestrator.get_status()
            agent_indicator = "[cyan]Planner[/cyan]" if not status["is_testing"] else "[magenta]Testing[/magenta]"
            user_input = console.input(f"\n[bold]{agent_indicator}[/bold] You > ").strip()

            if not user_input:
                continue

            # Check for commands
            if user_input.startswith("/"):
                if handle_command(user_input, orchestrator):
                    continue

            # Chat with the active agent
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response, metadata = orchestrator.chat(user_input)

            # Display response with agent label
            agent = metadata.get("agent", "planner")
            label = get_agent_label(agent)

            # Check for action transitions
            action = metadata.get("action", "")
            if action == "assessment_complete":
                result = metadata.get("assessment_result", {})
                verdict = result.get("verdict", "unknown")
                verdict_style = {
                    "mastered": "success",
                    "needs_review": "warning",
                    "not_ready": "bold red",
                }.get(verdict, "info")

                console.print(f"\n{label}:")
                console.print(response)
                console.print(
                    f"\n[{verdict_style}]Assessment verdict: {verdict.upper().replace('_', ' ')}[/{verdict_style}]"
                )
            else:
                # Show progress bar during testing
                if status["is_testing"]:
                    progress = _parse_progress(response)
                    if progress:
                        current, total = progress
                        filled = int(current / total * 10)
                        bar = f"[magenta]{'█' * filled}{'░' * (10 - filled)}[/magenta] {current}/{total}"
                        console.print(f"\n  {bar}")
                console.print(f"\n{label}: {response}")

        except KeyboardInterrupt:
            console.print("\n\n[dim]Use /quit to exit.[/dim]")
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            console.print("[dim]Try again or use /quit to exit.[/dim]")


if __name__ == "__main__":
    main()

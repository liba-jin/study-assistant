"""
Study Planner Agent — NANDA deployment server.

Wraps StudyPlannerAgent as a NANDA-compatible service.
Each message is processed statelessly (session state held in memory per agent_id).
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanda_adapter import NANDA
from agents.study_planner import StudyPlannerAgent

# In-memory sessions: agent_id -> StudyPlannerAgent instance
_sessions: dict[str, StudyPlannerAgent] = {}


def planner_agent(message: str) -> str:
    """
    Study Planner Agent handler for NANDA.

    Accepts natural language learning goals and returns a study plan
    or clarifying questions. Maintains session state per caller.

    Message format:
      Plain text — "I want to learn Python in 2 weeks"
      With session — "[session:abc123] I want to learn Python"
      Test results — "[results] topic=X verdict=mastered scores={...}"
    """
    # Parse optional session ID from message
    session_id = "default"
    content = message.strip()

    if content.startswith("[session:"):
        end = content.index("]")
        session_id = content[9:end]
        content = content[end + 1:].strip()

    # Get or create session
    if session_id not in _sessions:
        _sessions[session_id] = StudyPlannerAgent()

    agent = _sessions[session_id]

    try:
        response, action_signal = agent.chat(content)

        # If agent wants to invoke Test Agent, signal it
        if action_signal:
            topic = action_signal.get("topic", "")
            return (
                f"{response}\n\n"
                f"[INVOKE_TESTER] topic={topic}"
            )

        return response or "I'm ready to help you build a study plan. What would you like to learn?"

    except Exception as e:
        return f"Study Planner error: {str(e)}. Please try again."


if __name__ == "__main__":
    nanda = NANDA(planner_agent)
    nanda.start_server()

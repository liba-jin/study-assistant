"""
Test/Teacher Agent — NANDA deployment server.

Wraps TestTeacherAgent as a NANDA-compatible service.
Maintains session state per caller for multi-turn assessment.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanda_adapter import NANDA
from agents.test_teacher import TestTeacherAgent

# In-memory sessions: session_id -> TestTeacherAgent instance
_sessions: dict[str, TestTeacherAgent] = {}


def tester_agent(message: str) -> str:
    """
    Test/Teacher Agent handler for NANDA.

    Assesses understanding of any topic through adaptive quiz questions.
    Returns questions one at a time with feedback, then a final verdict.

    Message format:
      Start test  — "[session:abc123][start] topic=Transformers"
      Answer      — "[session:abc123] B"
      Direct test — "test me on cooking steak"
    """
    session_id = "default"
    content = message.strip()

    # Parse session ID
    if content.startswith("[session:"):
        end = content.index("]")
        session_id = content[9:end]
        content = content[end + 1:].strip()

    # Handle start command
    if content.startswith("[start]"):
        topic = content.replace("[start]", "").strip()
        if topic.startswith("topic="):
            topic = topic[6:].strip()

        agent = TestTeacherAgent()
        _sessions[session_id] = agent
        try:
            first_question = agent.start_assessment(topic)
            return first_question
        except Exception as e:
            return f"Error starting assessment: {str(e)}"

    # Handle direct "test me on X" messages (no session)
    if content.lower().startswith("test me on ") or content.lower().startswith("test "):
        topic = content.lower().replace("test me on ", "").replace("test ", "").strip()
        agent = TestTeacherAgent()
        _sessions[session_id] = agent
        try:
            return agent.start_assessment(topic)
        except Exception as e:
            return f"Error starting assessment: {str(e)}"

    # Continue existing session
    if session_id not in _sessions:
        return (
            "No active test session. Start one by sending:\n"
            "[start] topic=<your topic>\n"
            "Example: [start] topic=Transformer Architecture"
        )

    agent = _sessions[session_id]
    try:
        response, assessment_result = agent.chat(content)

        if assessment_result:
            # Assessment complete — clean up session
            del _sessions[session_id]
            verdict = assessment_result.get("verdict", "unknown")
            scores = assessment_result.get("sub_topic_scores", {})
            gaps = assessment_result.get("gaps", [])

            summary = (
                f"{response}\n\n"
                f"[ASSESSMENT_COMPLETE]\n"
                f"Verdict: {verdict.upper().replace('_', ' ')}\n"
                f"Scores: {json.dumps(scores, indent=2)}\n"
                f"Gaps: {', '.join(gaps) if gaps else 'None identified'}"
            )
            return summary

        return response or "Please answer the current question."

    except Exception as e:
        return f"Test Agent error: {str(e)}. Please try again."


if __name__ == "__main__":
    nanda = NANDA(tester_agent)
    nanda.start_server()

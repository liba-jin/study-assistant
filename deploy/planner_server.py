"""
Study Planner Agent — A2A deployment server.

Uses python_a2a directly so StudyPlannerAgent handles every message.
Sessions are keyed by A2A conversation_id for multi-turn conversations.
"""

import os
import sys
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_a2a import A2AServer, run_server, Message, TextContent, MessageRole

from agents.study_planner import StudyPlannerAgent

# In-memory sessions: conversation_id -> StudyPlannerAgent
_sessions: dict[str, StudyPlannerAgent] = {}


def _get_or_create_session(session_id: str) -> StudyPlannerAgent:
    if session_id not in _sessions:
        _sessions[session_id] = StudyPlannerAgent()
    return _sessions[session_id]


class PlannerA2AServer(A2AServer):
    """A2A server that routes every message to StudyPlannerAgent."""

    def handle_message(self, msg: Message) -> Message:
        conversation_id = msg.conversation_id or "default"
        user_text = msg.content.text.strip() if hasattr(msg.content, "text") else ""

        # Also support explicit [session:id] prefix for clients that use it
        if user_text.startswith("[session:"):
            end = user_text.index("]")
            conversation_id = user_text[9:end]
            user_text = user_text[end + 1:].strip()

        agent = _get_or_create_session(conversation_id)
        try:
            response_text, action_signal = agent.chat(user_text)
            # Surface test-agent invocation signal in the reply text
            if action_signal and action_signal.get("action") == "invoke_test_agent":
                topic = action_signal.get("topic", "")
                tester_url = os.getenv("TESTER_URL", "https://tester-yxtf.onrender.com")
                response_text = (
                    f"{response_text}\n\n"
                    f"[INVOKE_TESTER] topic={topic}\n"
                    f"Tester agent: {tester_url}"
                )
        except Exception as e:
            response_text = f"Planner error: {str(e)}"

        return Message(
            role=MessageRole.AGENT,
            content=TextContent(
                text=response_text
                or "Hi! Tell me your learning goal and I'll build a personalised study plan."
            ),
            parent_message_id=msg.message_id,
            conversation_id=conversation_id,
        )


def _register():
    """Register this agent with the NANDA registry if PUBLIC_URL is set."""
    agent_id = os.getenv("AGENT_ID", "studynet-planner")
    public_url = os.getenv("PUBLIC_URL", "")
    api_url = os.getenv("API_URL", public_url)
    registry_url = "https://chat.nanda-registry.com:6900"

    if not public_url:
        print("WARNING: PUBLIC_URL not set — skipping registry registration.")
        return

    try:
        resp = requests.post(
            f"{registry_url}/register",
            json={"agent_id": agent_id, "agent_url": public_url, "api_url": api_url},
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"Registered {agent_id} with NANDA registry")
        else:
            print(f"Registry registration failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"Registry registration error (non-fatal): {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "6000"))
    agent_id = os.getenv("AGENT_ID", "studynet-planner")
    public_url = os.getenv("PUBLIC_URL", "https://study-assistant-kjlw.onrender.com")
    print(f"Starting {agent_id} on port {port}")
    _register()
    run_server(PlannerA2AServer(url=public_url), host="0.0.0.0", port=port)

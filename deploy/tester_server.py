"""
Test/Teacher Agent — A2A deployment server.

Uses python_a2a directly so TestTeacherAgent handles every message.
Sessions are keyed by A2A conversation_id for multi-turn assessments.
"""

import os
import sys
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_a2a import A2AServer, run_server, Message, TextContent, MessageRole

from agents.test_teacher import TestTeacherAgent

# In-memory sessions: conversation_id -> TestTeacherAgent
_sessions: dict[str, TestTeacherAgent] = {}


class TesterA2AServer(A2AServer):
    """A2A server that routes every message to TestTeacherAgent."""

    def handle_message(self, msg: Message) -> Message:
        conversation_id = msg.conversation_id or "default"
        content = msg.content.text.strip() if hasattr(msg.content, "text") else ""

        # Support explicit [session:id] prefix
        if content.startswith("[session:"):
            end = content.index("]")
            conversation_id = content[9:end]
            content = content[end + 1:].strip()

        # Handle [start] topic=X  — begin a new assessment
        if content.startswith("[start]"):
            topic = content.replace("[start]", "").strip()
            if topic.startswith("topic="):
                topic = topic[6:].strip()

            agent = TestTeacherAgent()
            _sessions[conversation_id] = agent
            try:
                first_question = agent.start_assessment(topic)
                response_text = first_question
            except Exception as e:
                response_text = f"Error starting assessment: {str(e)}"

        # Handle direct "test me on X" shorthand
        elif content.lower().startswith("test me on ") or content.lower().startswith("test "):
            topic = (
                content.lower()
                .replace("test me on ", "")
                .replace("test ", "")
                .strip()
            )
            agent = TestTeacherAgent()
            _sessions[conversation_id] = agent
            try:
                response_text = agent.start_assessment(topic)
            except Exception as e:
                response_text = f"Error starting assessment: {str(e)}"

        # Continue existing session
        elif conversation_id in _sessions:
            agent = _sessions[conversation_id]
            try:
                response, assessment_result = agent.chat(content)

                if assessment_result:
                    # Assessment done — clean up
                    del _sessions[conversation_id]
                    verdict = assessment_result.get("verdict", "unknown")
                    scores = assessment_result.get("sub_topic_scores", {})
                    gaps = assessment_result.get("gaps", [])
                    response_text = (
                        f"{response}\n\n"
                        f"[ASSESSMENT_COMPLETE]\n"
                        f"Verdict: {verdict.upper().replace('_', ' ')}\n"
                        f"Scores: {json.dumps(scores, indent=2)}\n"
                        f"Gaps: {', '.join(gaps) if gaps else 'None identified'}"
                    )
                else:
                    response_text = response or "Please answer the current question."

            except Exception as e:
                response_text = f"Tester error: {str(e)}"

        else:
            response_text = (
                "No active assessment. Start one by sending:\n"
                "[start] topic=<your topic>\n\n"
                "Example: [start] topic=Transformer Architecture"
            )

        return Message(
            role=MessageRole.AGENT,
            content=TextContent(text=response_text),
            parent_message_id=msg.message_id,
            conversation_id=conversation_id,
        )


def _register():
    """Register this agent with the NANDA registry if PUBLIC_URL is set."""
    agent_id = os.getenv("AGENT_ID", "studynet-tester")
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
    agent_id = os.getenv("AGENT_ID", "studynet-tester")
    public_url = os.getenv("PUBLIC_URL", "https://tester-yxtf.onrender.com")
    print(f"Starting {agent_id} on port {port}")
    _register()
    run_server(TesterA2AServer(url=public_url), host="0.0.0.0", port=port)

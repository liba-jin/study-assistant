"""
Orchestrator — Routes between Study Planner and Test/Teacher agents.

Manages the inter-agent communication:
  - Study Planner can invoke Test Agent
  - Test Agent results feed back to Study Planner
  - User can directly invoke either agent
"""

from agents.study_planner import StudyPlannerAgent
from agents.test_teacher import TestTeacherAgent


class Orchestrator:
    """Coordinates the two-agent network."""

    def __init__(self):
        self.planner = StudyPlannerAgent()
        self.tester = TestTeacherAgent()
        self.active_agent = "planner"  # "planner" | "tester"
        self._pending_test_request = None

    @property
    def is_testing(self) -> bool:
        return self.active_agent == "tester"

    def chat(self, user_message: str) -> tuple[str, dict]:
        """
        Route user message to the active agent.

        Returns:
            (response_text, metadata)
            metadata includes: agent, action, assessment_result, visualization_path, etc.
        """
        metadata = {"agent": self.active_agent}

        if self.active_agent == "tester":
            return self._handle_tester_chat(user_message, metadata)
        else:
            return self._handle_planner_chat(user_message, metadata)

    def _handle_planner_chat(self, user_message: str, metadata: dict) -> tuple[str, dict]:
        """Handle chat with the Study Planner agent."""
        response_text, action_signal = self.planner.chat(user_message)

        if action_signal and action_signal.get("action") == "invoke_test_agent":
            # Planner wants to invoke the Test Agent
            topic = action_signal["topic"]
            plan_topic_id = action_signal.get("plan_topic_id")

            metadata["action"] = "starting_test"
            metadata["topic"] = topic

            # Start the test
            self.active_agent = "tester"
            first_question = self.tester.start_assessment(topic, plan_topic_id)

            return (
                f"{response_text}\n\n---\n\n"
                f"[Test Agent] Starting assessment on: {topic}\n\n"
                f"{first_question}"
            ), metadata

        return response_text, metadata

    def _handle_tester_chat(self, user_message: str, metadata: dict) -> tuple[str, dict]:
        """Handle chat with the Test/Teacher agent."""
        response_text, assessment_result = self.tester.chat(user_message)

        if assessment_result:
            # Assessment complete — switch back to planner
            metadata["action"] = "assessment_complete"
            metadata["assessment_result"] = assessment_result

            self.active_agent = "planner"

            # Feed results back to planner
            planner_response, _ = self.planner.receive_test_results(assessment_result)

            return (
                f"{response_text}\n\n---\n\n"
                f"[Study Planner] Adapting your plan based on results:\n\n"
                f"{planner_response}"
            ), metadata

        return response_text, metadata

    def start_test_directly(self, topic: str) -> tuple[str, dict]:
        """User directly invokes the Test Agent (not through planner)."""
        self.active_agent = "tester"
        first_question = self.tester.start_assessment(topic)
        metadata = {
            "agent": "tester",
            "action": "starting_test",
            "topic": topic,
        }
        return first_question, metadata

    def switch_to_planner(self) -> str:
        """Force switch back to planner (e.g., user wants to abort test)."""
        self.active_agent = "planner"
        return "Switched back to Study Planner. How can I help?"

    def get_status(self) -> dict:
        """Get current orchestrator status."""
        return {
            "active_agent": self.active_agent,
            "is_testing": self.is_testing,
            "tester_topic": self.tester.topic if self.is_testing else None,
        }

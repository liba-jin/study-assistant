"""
Test/Teacher Agent — Agent 2 of StudyNet.

Adaptively assesses understanding through Socratic dialogue.
Autonomous decisions:
  - Where to probe (detects blind spots)
  - How deep to go (adapts to answer quality)
  - Whether to challenge with counterexamples
  - When testing is sufficient
  - Mastery verdict
"""

import json
from groq import Groq

from config import MODEL, MAX_TOKENS, GROQ_API_KEY
from state import db
from tools.web_search import search_counterexample
from tools.visualization import generate_understanding_map, scores_to_rich_text

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are the Test/Teacher Agent in StudyNet, a personal learning assessment system.

Your job: Assess the user's understanding of a topic with a structured, focused test.

## CRITICAL RULES:
- Ask ONE specific question at a time. Wait for the answer before asking the next.
- NEVER ask follow-up questions on the same turn. One question per message, period.
- NEVER combine a quiz question with an open-ended explanation request.
- Keep the test SHORT: 5-8 questions total, then score and finish.

## Test format:
Use a MIX of question types (not all the same):
- Multiple choice (A/B/C/D) — for factual knowledge
- True/False — for common misconceptions
- Short answer — for key concepts (accept brief answers, 1-2 sentences is fine)
- Scenario-based — "What would you do if...?" for practical understanding

## How to run the test:

### Step 1: Identify 4-6 key sub-topics for this subject
### Step 2: Ask 5-8 questions covering those sub-topics (one per message)
### Step 3: After all questions answered, immediately:
1. Score each sub-topic (0.0 to 1.0)
2. Update the knowledge model
3. Generate visualization
4. Give a verdict with specific tips

## After EACH answer:
- Tell the user if they're right or wrong (briefly, 1 sentence)
- Give the correct answer if wrong (briefly)
- Then ask the NEXT question
- Do NOT probe deeper or ask follow-ups on the same topic

## Scoring criteria:
- 0.0-0.3: Wrong or no answer
- 0.3-0.5: Partially correct
- 0.5-0.7: Mostly correct but missing details
- 0.7-0.9: Correct with good understanding
- 0.9-1.0: Perfect, including edge cases

## Verdict criteria:
- mastered: ALL sub-topics >= 0.7
- needs_review: SOME sub-topics >= 0.5, but gaps remain
- not_ready: MOST sub-topics < 0.5

## Important:
- Be encouraging but HONEST
- Keep it moving — don't get stuck on one topic
- The user should feel tested, not interrogated
- Finish quickly and give actionable feedback
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_counterexample",
            "description": "Search for real-world counterexamples or edge cases to challenge the user's understanding. Use when the user makes a confident claim you want to test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The user's claim or explanation to find counterexamples for",
                    },
                    "topic": {
                        "type": "string",
                        "description": "The broader topic area",
                    },
                },
                "required": ["claim", "topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_knowledge_model",
            "description": "Update the user's persistent knowledge model with scores for each sub-topic tested. Call this after completing the assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Knowledge domain (e.g., 'machine_learning', 'cooking')",
                    },
                    "topic": {
                        "type": "string",
                        "description": "The main topic assessed",
                    },
                    "scores": {
                        "type": "object",
                        "description": "Dict of sub_topic to score (0.0 to 1.0)",
                    },
                },
                "required": ["domain", "topic", "scores"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_knowledge_history",
            "description": "Get past assessment scores for a topic to see if the user is improving or declining.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to get history for",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Generate an understanding map visualization showing scores per sub-topic. Call after scoring is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Main topic name",
                    },
                    "scores": {
                        "type": "object",
                        "description": "Dict of sub_topic to score (0.0 to 1.0)",
                    },
                },
                "required": ["topic", "scores"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_assessment",
            "description": "Record the final assessment result. Call this as the last step after scoring and visualization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                    },
                    "plan_topic_id": {
                        "type": "integer",
                        "description": "The plan_topic database ID (if from a plan)",
                    },
                    "sub_topic_scores": {
                        "type": "object",
                        "description": "Final scores per sub-topic",
                    },
                    "verdict": {
                        "type": "string",
                        "description": "mastered, needs_review, or not_ready",
                    },
                    "gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific knowledge gaps identified",
                    },
                },
                "required": ["topic", "sub_topic_scores", "verdict", "gaps"],
            },
        },
    },
]


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""

    if tool_name == "search_counterexample":
        results = search_counterexample(
            tool_input["claim"],
            tool_input["topic"],
        )
        return json.dumps(results, indent=2)

    elif tool_name == "update_knowledge_model":
        domain = tool_input["domain"]
        topic = tool_input["topic"]
        scores = tool_input.get("scores", {})
        for sub_topic, score in scores.items():
            db.update_knowledge(domain, topic, sub_topic, float(score))
        return json.dumps({"success": True, "updated_count": len(scores)})

    elif tool_name == "get_knowledge_history":
        knowledge = db.get_knowledge(tool_input["topic"])
        return json.dumps(knowledge, indent=2, default=str)

    elif tool_name == "generate_visualization":
        topic = tool_input["topic"]
        scores = {k: float(v) for k, v in tool_input.get("scores", {}).items()}
        image_path = generate_understanding_map(topic, scores)
        return json.dumps({
            "success": True,
            "image_path": image_path,
            "rich_text": scores_to_rich_text(topic, scores, "", []),
        })

    elif tool_name == "finalize_assessment":
        scores = {k: float(v) for k, v in tool_input.get("sub_topic_scores", {}).items()}
        assessment_id = db.save_assessment(
            plan_topic_id=tool_input.get("plan_topic_id"),
            topic=tool_input["topic"],
            sub_topic_scores=scores,
            verdict=tool_input["verdict"],
            gaps=tool_input.get("gaps", []),
        )
        return json.dumps({
            "action": "assessment_complete",
            "assessment_id": assessment_id,
            "topic": tool_input["topic"],
            "verdict": tool_input["verdict"],
            "sub_topic_scores": scores,
            "gaps": tool_input.get("gaps", []),
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _clean_message(msg) -> dict:
    """Clean a message dict to only include fields Groq accepts."""
    cleaned = {"role": msg.get("role", "assistant")}
    if msg.get("content") is not None:
        cleaned["content"] = msg["content"]
    if msg.get("tool_calls"):
        cleaned["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }
            for tc in msg["tool_calls"]
        ]
    if msg.get("tool_call_id"):
        cleaned["tool_call_id"] = msg["tool_call_id"]
    return cleaned


MAX_HISTORY = 16  # Slightly more for test agent (needs full Q&A context)


def _trim_history(messages: list[dict]) -> list[dict]:
    """Keep system message + last MAX_HISTORY messages to stay within token limits."""
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) > MAX_HISTORY:
        non_system = non_system[-MAX_HISTORY:]
    return system + non_system


class TestTeacherAgent:
    """Test/Teacher Agent with adaptive Socratic dialogue."""

    def __init__(self):
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.assessment_result: dict | None = None
        self.topic: str = ""
        self.plan_topic_id: int | None = None

    def start_assessment(self, topic: str, plan_topic_id: int = None) -> str:
        """
        Begin a new assessment for a topic.
        Returns the first question/prompt for the user.
        """
        self.topic = topic
        self.plan_topic_id = plan_topic_id
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.assessment_result = None

        # Check if we have prior knowledge to reference
        history = db.get_knowledge(topic)
        history_context = ""
        if history:
            history_context = (
                f"\nPrior knowledge model for this topic:\n"
                + json.dumps(history, indent=2, default=str)
                + "\nUse this to focus on previously weak areas.\n"
            )

        init_message = (
            f"Topic to assess: {topic}\n"
            f"Plan topic ID: {plan_topic_id}\n"
            f"{history_context}\n"
            f"RESPOND WITH ONLY YOUR FIRST QUESTION. Nothing else.\n"
            f"Format: 'Question 1 of N: [multiple choice question with A/B/C/D options]'\n"
            f"No preamble. No instructions. No explanation. Just the question."
        )

        return self._chat_internal(init_message)

    def chat(self, user_message: str) -> tuple[str, dict | None]:
        """
        Continue the assessment conversation.

        Returns:
            (response_text, assessment_result)
            assessment_result is non-None when the assessment is complete.
        """
        response_text = self._chat_internal(user_message)
        return response_text, self.assessment_result

    def _chat_internal(self, message: str) -> str:
        """Internal chat that handles the tool use loop."""
        self.messages.append({"role": "user", "content": message})

        while True:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=_trim_history(self.messages),
                tools=TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            self.messages.append(_clean_message(msg.model_dump()))

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func = tool_call.function
                    try:
                        args = json.loads(func.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    # Inject plan_topic_id for finalize_assessment
                    if func.name == "finalize_assessment" and self.plan_topic_id:
                        args["plan_topic_id"] = self.plan_topic_id

                    result = handle_tool_call(func.name, args)

                    # Check if assessment is complete
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, dict) and parsed.get("action") == "assessment_complete":
                            self.assessment_result = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                # Continue loop for model to process tool results
                continue

            # No tool calls — return text
            return msg.content or ""

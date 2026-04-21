"""
Study Planner Agent — Agent 1 of StudyNet.

Transforms any learning goal (vague or structured) into an adaptive weekly plan.
Autonomous decisions:
  - Is the goal specific enough? If not, interviews the user
  - What type of resource for each topic?
  - When to trigger assessment via Test Agent
  - How to adapt plan based on test results
"""

import json
from groq import Groq

from config import MODEL, MAX_TOKENS, GROQ_API_KEY
from state import db
from tools.youtube_search import search_courses
from tools.web_search import search_web

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are the Study Planner Agent in StudyNet, a personal learning system.

Your job: Take any learning goal (vague or structured) and create an actionable, adaptive study plan.

## How you operate:

### When the goal is VAGUE (e.g., "I want to learn cooking"):
1. Interview the user to pin down what they actually need:
   - What specific area? (e.g., daily meals vs baking vs a specific cuisine)
   - What do they already know?
   - What's the end goal? (eat healthier, career, hobby?)
   - How much time do they have? Hours per day/week?
2. YOU decide when you have enough info — don't over-ask
3. Then generate the plan

### When the goal is STRUCTURED (e.g., a syllabus or topic list):
- Parse it directly into a weekly/topic plan
- Still search for the best resources

### When generating a plan:
- Break the goal into sequential topics with clear learning objectives
- For each topic, use your tools to find the best resources:
  - Use search_courses for video tutorials (YouTube)
  - Use search_web for articles, MOOCs, documentation
- Decide the right resource TYPE for each topic (some topics need videos, others need docs)

### When you receive test results:
- Analyze the mastery verdict and sub-topic scores
- DECIDE how to adapt:
  - mastered → advance to next topic
  - needs_review → add supplementary resources, keep topic active
  - not_ready → restructure: break topic into smaller pieces, find alternative explanations

### When to invoke the Test Agent:
- After the user indicates they've studied a topic
- You can suggest assessment when you think they're ready
- Use invoke_test_agent tool with the topic name
- NEVER generate quiz questions yourself — that is the Test Agent's job
- NEVER mix quiz questions with your plan or responses

## Important:
- Always use tools to find REAL resources — never hallucinate URLs
- Be encouraging but honest about gaps
- Adapt the plan based on the user's actual progress, not a fixed schedule
- You are a PLANNER, not a tester. Do not quiz the user yourself.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_courses",
            "description": "Search YouTube for video tutorials and courses on a topic. Use for topics that benefit from visual/video explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for courses/tutorials",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for articles, MOOCs, documentation, and learning resources. Use for topics that need text-based resources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_plan",
            "description": "Save the generated study plan to the database. Call this after creating or updating a plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The original learning goal",
                    },
                    "refined_goal": {
                        "type": "string",
                        "description": "The clarified goal after interview (if applicable)",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Time commitment (e.g., '2 weeks', '1 month')",
                    },
                    "topics": {
                        "type": "array",
                        "description": "Ordered list of topics with objectives and resources",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "objectives": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "resources": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "url": {"type": "string"},
                                            "type": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "required": ["goal", "topics"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_plan",
            "description": "Get the current active study plan with all topics and their status.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_topic_status",
            "description": "Update the status of a plan topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_id": {"type": "integer", "description": "The topic ID"},
                    "status": {
                        "type": "string",
                        "description": "New status: pending, in_progress, assessed, or completed",
                    },
                },
                "required": ["topic_id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invoke_test_agent",
            "description": "Invoke the Test/Teacher Agent to assess user understanding of a topic. Use when the user is ready to be tested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to test the user on",
                    },
                    "plan_topic_id": {
                        "type": "integer",
                        "description": "The plan_topic database ID (if from a plan)",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adapt_plan",
            "description": "Restructure the plan based on test results. Add new topics, reorder, or add resources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_id": {"type": "integer", "description": "The plan ID"},
                    "action": {
                        "type": "string",
                        "description": "Action: add_review_topic, add_resources, reorder, or extend",
                    },
                    "after_topic_id": {
                        "type": "integer",
                        "description": "Insert after this topic (for add_review_topic)",
                    },
                    "topic": {
                        "type": "string",
                        "description": "New topic name (for add_review_topic)",
                    },
                    "objectives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Objectives for new topic",
                    },
                },
                "required": ["plan_id", "action"],
            },
        },
    },
]


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""

    if tool_name == "search_courses":
        results = search_courses(
            tool_input["query"],
            tool_input.get("max_results", 5),
        )
        return json.dumps(results, indent=2)

    elif tool_name == "search_web":
        results = search_web(
            tool_input["query"],
            tool_input.get("max_results", 5),
        )
        return json.dumps(results, indent=2)

    elif tool_name == "save_plan":
        plan_id = db.create_plan(
            goal=tool_input["goal"],
            refined_goal=tool_input.get("refined_goal"),
            timeframe=tool_input.get("timeframe"),
        )
        for i, topic_data in enumerate(tool_input["topics"]):
            db.add_plan_topic(
                plan_id=plan_id,
                sequence=i + 1,
                topic=topic_data["topic"],
                objectives=topic_data.get("objectives", []),
                resources=topic_data.get("resources", []),
            )
        return json.dumps({"success": True, "plan_id": plan_id})

    elif tool_name == "get_current_plan":
        plan = db.get_active_plan()
        if not plan:
            return json.dumps({"error": "No active plan found"})
        topics = db.get_plan_topics(plan["id"])
        return json.dumps({"plan": plan, "topics": topics}, indent=2, default=str)

    elif tool_name == "update_topic_status":
        db.update_topic_status(tool_input["topic_id"], tool_input["status"])
        return json.dumps({"success": True})

    elif tool_name == "invoke_test_agent":
        return json.dumps({
            "action": "invoke_test_agent",
            "topic": tool_input["topic"],
            "plan_topic_id": tool_input.get("plan_topic_id"),
        })

    elif tool_name == "adapt_plan":
        action = tool_input["action"]
        plan_id = tool_input["plan_id"]
        if action == "add_review_topic":
            topic_id = db.add_topic_after(
                plan_id=plan_id,
                after_sequence=tool_input.get("after_topic_id", 0),
                topic=tool_input.get("topic", "Review"),
                objectives=tool_input.get("objectives", []),
            )
            return json.dumps({"success": True, "new_topic_id": topic_id})
        return json.dumps({"success": True, "action": action})

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


MAX_HISTORY = 12  # Max non-system messages to keep (prevents token limit errors)


def _trim_history(messages: list[dict]) -> list[dict]:
    """Keep system message + last MAX_HISTORY messages to stay within token limits."""
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) > MAX_HISTORY:
        non_system = non_system[-MAX_HISTORY:]
    return system + non_system


class StudyPlannerAgent:
    """Study Planner Agent with persistent conversation and tool use."""

    def __init__(self):
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def chat(self, user_message: str) -> tuple[str, dict | None]:
        """
        Send a message to the Study Planner.

        Returns:
            (response_text, action_signal)
            action_signal is non-None when the agent wants to invoke another agent.
        """
        self.messages.append({"role": "user", "content": user_message})

        action_signal = None

        while True:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=_trim_history(self.messages),
                tools=TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # Append assistant message to history
            self.messages.append(_clean_message(message.model_dump()))

            # Check for tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    func = tool_call.function
                    try:
                        args = json.loads(func.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    result = handle_tool_call(func.name, args)

                    # Check for inter-agent signals
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, dict) and parsed.get("action") == "invoke_test_agent":
                            action_signal = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                # If test agent invocation was requested, get final response
                if action_signal:
                    final_response = client.chat.completions.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        messages=_trim_history(self.messages),
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                    final_msg = final_response.choices[0].message
                    self.messages.append(_clean_message(final_msg.model_dump()))
                    return final_msg.content or "", action_signal

                # Otherwise continue the tool use loop
                continue

            # No tool calls — return text
            return message.content or "", action_signal

    def receive_test_results(self, results: dict):
        """Feed test results back into the planner so it can adapt."""
        msg = (
            f"The Test Agent just assessed the user. Here are the results:\n"
            f"Topic: {results.get('topic', 'unknown')}\n"
            f"Verdict: {results.get('verdict', 'unknown')}\n"
            f"Sub-topic scores: {json.dumps(results.get('sub_topic_scores', {}))}\n"
            f"Gaps identified: {json.dumps(results.get('gaps', []))}\n\n"
            f"Based on these results, decide how to adapt the study plan. "
            f"If the verdict is 'mastered', advance to the next topic. "
            f"If 'needs_review' or 'not_ready', restructure the plan accordingly."
        )
        return self.chat(msg)

"""SQLite database layer for StudyNet state management."""

import json
import sqlite3
from datetime import datetime
from typing import Any, Optional

from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal TEXT NOT NULL,
            refined_goal TEXT,
            timeframe TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active'
        );

        CREATE TABLE IF NOT EXISTS plan_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id INTEGER REFERENCES plans(id),
            sequence INTEGER,
            topic TEXT NOT NULL,
            objectives TEXT,
            resources TEXT,
            status TEXT DEFAULT 'pending'
        );

        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_topic_id INTEGER REFERENCES plan_topics(id),
            topic TEXT NOT NULL,
            sub_topic_scores TEXT,
            verdict TEXT,
            gaps TEXT,
            conversation_log TEXT,
            assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS knowledge_model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT,
            topic TEXT NOT NULL,
            sub_topic TEXT NOT NULL,
            score REAL DEFAULT 0.0,
            trend TEXT DEFAULT 'new',
            last_tested TIMESTAMP,
            times_tested INTEGER DEFAULT 0,
            UNIQUE(domain, topic, sub_topic)
        );
    """)
    conn.commit()
    conn.close()


# --- Plan operations ---

def create_plan(goal: str, refined_goal: str = None, timeframe: str = None) -> int:
    """Create a new study plan. Returns plan ID."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO plans (goal, refined_goal, timeframe) VALUES (?, ?, ?)",
        (goal, refined_goal, timeframe),
    )
    plan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return plan_id


def get_plan(plan_id: int) -> Optional[dict]:
    """Get a plan by ID."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_active_plan() -> Optional[dict]:
    """Get the currently active plan."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM plans WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_plan_status(plan_id: int, status: str):
    """Update plan status (active/completed/paused)."""
    conn = get_connection()
    conn.execute("UPDATE plans SET status = ? WHERE id = ?", (status, plan_id))
    conn.commit()
    conn.close()


def update_plan_refined_goal(plan_id: int, refined_goal: str, timeframe: str = None):
    """Update plan after goal clarification interview."""
    conn = get_connection()
    conn.execute(
        "UPDATE plans SET refined_goal = ?, timeframe = ? WHERE id = ?",
        (refined_goal, timeframe, plan_id),
    )
    conn.commit()
    conn.close()


# --- Plan topic operations ---

def add_plan_topic(
    plan_id: int, sequence: int, topic: str,
    objectives: list = None, resources: list = None,
) -> int:
    """Add a topic to a plan. Returns topic ID."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO plan_topics (plan_id, sequence, topic, objectives, resources) VALUES (?, ?, ?, ?, ?)",
        (plan_id, sequence, topic, json.dumps(objectives or []), json.dumps(resources or [])),
    )
    topic_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return topic_id


def get_plan_topics(plan_id: int) -> list[dict]:
    """Get all topics for a plan, ordered by sequence."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM plan_topics WHERE plan_id = ? ORDER BY sequence", (plan_id,)
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["objectives"] = json.loads(d["objectives"]) if d["objectives"] else []
        d["resources"] = json.loads(d["resources"]) if d["resources"] else []
        result.append(d)
    return result


def update_topic_status(topic_id: int, status: str):
    """Update topic status (pending/in_progress/assessed/completed)."""
    conn = get_connection()
    conn.execute("UPDATE plan_topics SET status = ? WHERE id = ?", (status, topic_id))
    conn.commit()
    conn.close()


def update_topic_resources(topic_id: int, resources: list):
    """Update resources for a topic."""
    conn = get_connection()
    conn.execute(
        "UPDATE plan_topics SET resources = ? WHERE id = ?",
        (json.dumps(resources), topic_id),
    )
    conn.commit()
    conn.close()


def add_topic_after(plan_id: int, after_sequence: int, topic: str, objectives: list = None) -> int:
    """Insert a new topic after a given sequence number (for plan restructuring)."""
    conn = get_connection()
    # Shift subsequent topics
    conn.execute(
        "UPDATE plan_topics SET sequence = sequence + 1 WHERE plan_id = ? AND sequence > ?",
        (plan_id, after_sequence),
    )
    cursor = conn.execute(
        "INSERT INTO plan_topics (plan_id, sequence, topic, objectives, resources) VALUES (?, ?, ?, ?, ?)",
        (plan_id, after_sequence + 1, topic, json.dumps(objectives or []), json.dumps([])),
    )
    topic_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return topic_id


# --- Assessment operations ---

def save_assessment(
    plan_topic_id: int, topic: str, sub_topic_scores: dict,
    verdict: str, gaps: list, conversation_log: str = "",
) -> int:
    """Save an assessment result. Returns assessment ID."""
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO assessments
           (plan_topic_id, topic, sub_topic_scores, verdict, gaps, conversation_log)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            plan_topic_id, topic, json.dumps(sub_topic_scores),
            verdict, json.dumps(gaps), conversation_log,
        ),
    )
    assessment_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return assessment_id


def get_assessments_for_topic(plan_topic_id: int) -> list[dict]:
    """Get all assessments for a plan topic."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM assessments WHERE plan_topic_id = ? ORDER BY assessed_at",
        (plan_topic_id,),
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["sub_topic_scores"] = json.loads(d["sub_topic_scores"]) if d["sub_topic_scores"] else {}
        d["gaps"] = json.loads(d["gaps"]) if d["gaps"] else []
        result.append(d)
    return result


# --- Knowledge model operations ---

def update_knowledge(domain: str, topic: str, sub_topic: str, score: float):
    """Update or insert a knowledge model entry."""
    conn = get_connection()
    existing = conn.execute(
        "SELECT * FROM knowledge_model WHERE domain = ? AND topic = ? AND sub_topic = ?",
        (domain, topic, sub_topic),
    ).fetchone()

    now = datetime.now().isoformat()

    if existing:
        old_score = existing["score"]
        times = existing["times_tested"] + 1
        if score > old_score:
            trend = "improving"
        elif score < old_score:
            trend = "declining"
        else:
            trend = "stable"

        conn.execute(
            """UPDATE knowledge_model
               SET score = ?, trend = ?, last_tested = ?, times_tested = ?
               WHERE domain = ? AND topic = ? AND sub_topic = ?""",
            (score, trend, now, times, domain, topic, sub_topic),
        )
    else:
        conn.execute(
            """INSERT INTO knowledge_model
               (domain, topic, sub_topic, score, trend, last_tested, times_tested)
               VALUES (?, ?, ?, ?, 'new', ?, 1)""",
            (domain, topic, sub_topic, score, now),
        )

    conn.commit()
    conn.close()


def get_knowledge(topic: str, sub_topic: str = None) -> list[dict]:
    """Get knowledge model entries for a topic (optionally filtered by sub-topic)."""
    conn = get_connection()
    if sub_topic:
        rows = conn.execute(
            "SELECT * FROM knowledge_model WHERE topic = ? AND sub_topic = ?",
            (topic, sub_topic),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM knowledge_model WHERE topic = ?", (topic,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_knowledge(domain: str = None) -> list[dict]:
    """Get all knowledge model entries, optionally filtered by domain."""
    conn = get_connection()
    if domain:
        rows = conn.execute(
            "SELECT * FROM knowledge_model WHERE domain = ? ORDER BY topic, sub_topic",
            (domain,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM knowledge_model ORDER BY domain, topic, sub_topic"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize DB on import
init_db()

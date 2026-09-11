"""
AI Career-Mentor Agent
------------------------------------------------------
An agentic AI system with three cooperating agents:

  1. SkillAssessorAgent   -> evaluates the student's current skills
  2. PathRecommenderAgent -> plans a personalized learning path
  3. ProgressTrackerAgent -> tracks completion and gives next steps

The agents communicate through a shared "AgentState" object,
mimicking how multi-agent orchestration works in real
agentic-AI frameworks (e.g. watsonx Orchestrate / LangGraph).

Persistence: progress is saved to progress.json so the agent
"remembers" the student across runs (a basic form of memory).
"""

import json
import os
from datetime import date

DB_FILE = os.path.join(os.path.dirname(__file__), "progress.json")

# ---------------------------------------------------------------
# Knowledge base: career goal -> ordered list of skills/courses
# In a production build this would come from watsonx.ai / a
# vector DB of course catalogs instead of a static dict.
# ---------------------------------------------------------------
LEARNING_PATHS = {
    "data scientist": [
        "Python Programming",
        "Statistics & Probability",
        "Data Wrangling (Pandas)",
        "Machine Learning Fundamentals",
        "Deep Learning (TensorFlow/PyTorch)",
        "SQL & Databases",
        "Data Visualization",
        "MLOps Basics",
    ],
    "web developer": [
        "HTML/CSS Fundamentals",
        "JavaScript Essentials",
        "React or Vue Framework",
        "REST API Design",
        "Node.js Backend",
        "Databases (SQL/NoSQL)",
        "Git & Deployment",
    ],
    "cloud engineer": [
        "Linux Fundamentals",
        "Networking Basics",
        "IBM Cloud / AWS / Azure Core Services",
        "Containers (Docker)",
        "Kubernetes",
        "CI/CD Pipelines",
        "Cloud Security Basics",
    ],
    "ai engineer": [
        "Python Programming",
        "Linear Algebra & Calculus",
        "Machine Learning Fundamentals",
        "Deep Learning",
        "NLP & LLMs",
        "Prompt Engineering",
        "watsonx.ai / Model Deployment",
    ],
}


class AgentState:
    """Shared memory object passed between agents."""

    def __init__(self, student_name):
        self.student_name = student_name
        self.known_skills = []
        self.target_role = None
        self.learning_path = []
        self.completed = []
        self.log = []

    def note(self, agent, message):
        entry = f"[{agent}] {message}"
        self.log.append(entry)
        print(entry)


class SkillAssessorAgent:
    """Agent 1: figures out what the student already knows."""

    def run(self, state: AgentState, known_skills, target_role):
        state.known_skills = [s.strip().lower() for s in known_skills]
        state.target_role = target_role.strip().lower()
        state.note(
            "SkillAssessorAgent",
            f"Assessed {state.student_name}: knows {len(state.known_skills)} "
            f"skills, target role = '{state.target_role}'.",
        )
        return state


class PathRecommenderAgent:
    """Agent 2: builds a personalized learning path, skipping
    skills the student already has."""

    def run(self, state: AgentState):
        role = state.target_role
        if role not in LEARNING_PATHS:
            state.note(
                "PathRecommenderAgent",
                f"No preset path for '{role}'. Defaulting to 'ai engineer'.",
            )
            role = "ai engineer"

        full_path = LEARNING_PATHS[role]
        remaining = [
            skill for skill in full_path
            if skill.lower() not in state.known_skills
        ]
        state.learning_path = remaining
        state.note(
            "PathRecommenderAgent",
            f"Recommended {len(remaining)} learning steps for '{state.target_role}' "
            f"(skipped {len(full_path) - len(remaining)} already-known skills).",
        )
        return state


class ProgressTrackerAgent:
    """Agent 3: saves/loads progress and reports next action."""

    def load(self, state: AgentState):
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "r") as f:
                db = json.load(f)
            record = db.get(state.student_name)
            if record:
                state.completed = record.get("completed", [])
                state.note(
                    "ProgressTrackerAgent",
                    f"Loaded existing record: {len(state.completed)} steps already completed.",
                )
        return state

    def save(self, state: AgentState):
        db = {}
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "r") as f:
                db = json.load(f)
        db[state.student_name] = {
            "target_role": state.target_role,
            "learning_path": state.learning_path,
            "completed": state.completed,
            "last_updated": str(date.today()),
        }
        with open(DB_FILE, "w") as f:
            json.dump(db, f, indent=2)

    def mark_complete(self, state: AgentState, skill):
        if skill in state.learning_path and skill not in state.completed:
            state.completed.append(skill)
            state.note("ProgressTrackerAgent", f"Marked '{skill}' as completed.")
        self.save(state)
        return state

    def report(self, state: AgentState):
        pending = [s for s in state.learning_path if s not in state.completed]
        pct = (
            round(100 * len(state.completed) / len(state.learning_path))
            if state.learning_path else 100
        )
        state.note(
            "ProgressTrackerAgent",
            f"Progress: {len(state.completed)}/{len(state.learning_path)} "
            f"steps done ({pct}%).",
        )
        next_step = pending[0] if pending else "All steps complete! 🎉"
        state.note("ProgressTrackerAgent", f"Next recommended step: {next_step}")
        return pct, next_step


def run_career_mentor(student_name, known_skills, target_role, newly_completed=None):
    """Orchestrator: runs the 3 agents in sequence (an agentic pipeline)."""
    print(f"\n{'='*60}\nAI CAREER-MENTOR AGENT — session for {student_name}\n{'='*60}")

    state = AgentState(student_name)

    assessor = SkillAssessorAgent()
    recommender = PathRecommenderAgent()
    tracker = ProgressTrackerAgent()

    state = assessor.run(state, known_skills, target_role)
    state = tracker.load(state)
    state = recommender.run(state)

    if newly_completed:
        for skill in newly_completed:
            state = tracker.mark_complete(state, skill)

    pct, next_step = tracker.report(state)

    print(f"\n--- Personalized Learning Path for {student_name} ---")
    for i, skill in enumerate(state.learning_path, 1):
        status = "[DONE]" if skill in state.completed else "[ ]"
        print(f" {i}. {status} {skill}")

    print(f"\nCompletion: {pct}%  |  Next step -> {next_step}\n")
    return state


if __name__ == "__main__":
    # ---- Demo run 1: new student ----
    run_career_mentor(
        student_name="Priya",
        known_skills=["HTML/CSS Fundamentals", "Python Programming"],
        target_role="Data Scientist",
    )

    # ---- Demo run 2: same student returns later, completes some steps ----
    run_career_mentor(
        student_name="Priya",
        known_skills=["HTML/CSS Fundamentals", "Python Programming"],
        target_role="Data Scientist",
        newly_completed=["Statistics & Probability", "Data Wrangling (Pandas)"],
    )

    # ---- Demo run 3: a different student, different goal ----
    run_career_mentor(
        student_name="Arjun",
        known_skills=["Linux Fundamentals"],
        target_role="Cloud Engineer",
    )

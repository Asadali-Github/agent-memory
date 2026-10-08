"""Custom long-horizon task (Mode B: memory use DURING execution — carries the
agent claim).

"Project-Tracker Assistant": the agent observes a long stream of project
updates (facts planted: deadlines, budgets, owners, statuses) mixed with
distractor chatter. At interleaved ACTION steps it must answer operational
queries whose evidence lies many steps back. Facts get UPDATED over time
(newest value is correct), probing update-on-conflict (M3) and recency
weighting (M4) as well as plain retrieval (M1) and recency windows (B).

Fully deterministic from seed; grading is programmatic (normalised match on
the planted value) — no judge needed for Mode B.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from agentmem.memory.base import Turn

PROJECTS = ["Falcon", "Nimbus", "Quasar", "Zephyr", "Basilisk", "Aurora",
            "Meridian", "Sable", "Krypton", "Vortex"]
OWNERS = ["Priya Sharma", "Tom Becker", "Lena Fischer", "Marco Rossi",
          "Aisha Khan", "David Osei", "Ingrid Larsen", "Omar Haddad"]
STATUSES = ["on track", "at risk", "blocked", "ahead of schedule"]
DISTRACTORS = [
    "Reminder: the kitchen on floor 3 is closed for cleaning today.",
    "The all-hands meeting moved to the main auditorium.",
    "IT will run scheduled maintenance over the weekend.",
    "New starters should complete security training by Friday.",
    "The travel policy document was reformatted, no content changes.",
    "Facilities replaced the printers on floor 2.",
]

ATTRS = ("deadline", "budget", "owner", "status")


@dataclass
class ModeBQuery:
    step: int                 # step index at which the query is asked
    question: str
    answer: str               # gold = latest planted value at that step
    evidence_steps: list[int] # steps where the current gold value was stated
    attr: str
    project: str
    distance: int             # steps since the evidence was planted


@dataclass
class ModeBEpisode:
    episode_id: str
    turns: list[Turn]         # observation stream (chronological)
    queries: list[ModeBQuery]
    metadata: dict = field(default_factory=dict)


def _value(rng: random.Random, attr: str) -> str:
    if attr == "deadline":
        return f"{rng.randint(1, 28)} {rng.choice(['July', 'August', 'September', 'October'])} 2026"
    if attr == "budget":
        return f"£{rng.randint(10, 500)}k"
    if attr == "owner":
        return rng.choice(OWNERS)
    return rng.choice(STATUSES)


def _statement(project: str, attr: str, value: str) -> str:
    return {
        "deadline": f"Update: project {project}'s deadline is now {value}.",
        "budget": f"Update: project {project}'s budget is set to {value}.",
        "owner": f"Update: {value} is now the owner of project {project}.",
        "status": f"Update: project {project} is currently {value}.",
    }[attr]


def _question(project: str, attr: str) -> str:
    return {
        "deadline": f"What is the current deadline of project {project}?",
        "budget": f"What is the current budget of project {project}?",
        "owner": f"Who currently owns project {project}?",
        "status": f"What is the current status of project {project}?",
    }[attr]


def generate_episode(
    episode_id: str,
    seed: int,
    n_steps: int = 120,
    n_projects: int = 6,
    n_queries: int = 10,
    update_prob: float = 0.25,
    distractor_prob: float = 0.35,
    min_evidence_gap: int = 20,
) -> ModeBEpisode:
    rng = random.Random(seed)
    projects = rng.sample(PROJECTS, n_projects)
    # state[(project, attr)] = (value, [steps where current value was stated])
    state: dict[tuple[str, str], tuple[str, list[int]]] = {}
    turns: list[Turn] = []
    step = 0

    def plant(project: str, attr: str) -> None:
        nonlocal step
        value = _value(rng, attr)
        state[(project, attr)] = (value, [step])
        turns.append(Turn(role="system", content=_statement(project, attr, value),
                          step=step, metadata={"kind": "fact", "project": project,
                                               "attr": attr, "value": value}))
        step += 1

    # Seed every project/attr once up front so all queries are answerable.
    for p in projects:
        for a in ATTRS:
            plant(p, a)

    # Observation stream: distractors + fact updates.
    while step < n_steps:
        if rng.random() < distractor_prob:
            turns.append(Turn(role="system", content=rng.choice(DISTRACTORS),
                              step=step, metadata={"kind": "distractor"}))
            step += 1
        elif rng.random() < update_prob:
            plant(rng.choice(projects), rng.choice(ATTRS))
        else:
            p = rng.choice(projects)
            turns.append(Turn(role="user",
                              content=f"Thanks for the update on {p}, keep me posted.",
                              step=step, metadata={"kind": "chatter"}))
            step += 1

    # Queries at deterministic late positions (last 40% of the episode).
    queries: list[ModeBQuery] = []
    keys = sorted(state.keys())
    rng.shuffle(keys)
    q_positions = sorted(rng.sample(range(int(n_steps * 0.6), n_steps),
                                    min(n_queries, n_steps - int(n_steps * 0.6))))
    for pos, (project, attr) in zip(q_positions, keys):
        value, ev_steps = state[(project, attr)]
        # Gold = latest value planted BEFORE the query position.
        valid = [s for s in ev_steps if s < pos]
        if not valid:
            continue
        # Validity control: the evidence for the current gold value must lie at
        # least `min_evidence_gap` steps back, so the baseline recency window is
        # genuinely memory-limited on every retained query (not trivially caught
        # by keeping the last few turns). Queries violating the floor are dropped.
        if pos - max(valid) < min_evidence_gap:
            continue
        queries.append(ModeBQuery(
            step=pos, question=_question(project, attr), answer=value,
            evidence_steps=valid, attr=attr, project=project,
            distance=pos - max(valid),
        ))

    return ModeBEpisode(
        episode_id=episode_id, turns=turns, queries=queries,
        metadata={"seed": seed, "n_steps": n_steps, "n_projects": n_projects},
    )


def generate_episodes(n_episodes: int = 10, seed: int = 1, **kwargs) -> list[ModeBEpisode]:
    return [
        generate_episode(episode_id=f"ep{seed}_{i:03d}", seed=seed * 10_000 + i, **kwargs)
        for i in range(n_episodes)
    ]


def grade_answer(predicted: str, gold: str) -> bool:
    """Normalised containment: gold value must appear in the prediction."""
    norm = lambda s: " ".join(s.lower().replace("£", "").split())  # noqa: E731
    return norm(gold) in norm(predicted)

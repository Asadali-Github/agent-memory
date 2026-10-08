#!/usr/bin/env python
"""Entry point. One command per (task, condition, seed) cell; resumable.

Examples:
  python scripts/run_experiment.py --task custom_ds --condition m1 --seed 1
  python scripts/run_experiment.py --task longmemeval_s --condition b --seed 1 --limit 20
  python scripts/run_experiment.py --task custom_ds --all-conditions --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from agentmem.harness.runner import run_mode_a, run_mode_b  # noqa: E402
from agentmem.llm.client import make_llm  # noqa: E402
from agentmem.logging.trajectory import TrajectoryWriter  # noqa: E402
from agentmem.memory import ALL_ARMS, CONDITIONS  # noqa: E402
from agentmem.tasks.custom_ds import generate_episodes  # noqa: E402
from agentmem.tasks.longmemeval import load_longmemeval  # noqa: E402
from agentmem.utils.manifest import write_manifest  # noqa: E402


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["longmemeval_s", "custom_ds"])
    # --condition accepts control arms (e.g. b_reversed); --all-conditions does
    # NOT, so the registered main grid stays exactly the five pre-specified arms.
    ap.add_argument("--condition", choices=list(ALL_ARMS))
    ap.add_argument("--all-conditions", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="cap items/episodes (pilot)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default=None, help="results JSONL (default per task)")
    args = ap.parse_args()
    if not args.condition and not args.all_conditions:
        ap.error("need --condition or --all-conditions")

    load_dotenv(ROOT / ".env")
    config = load_yaml(ROOT / args.config)
    task_cfg = load_yaml(ROOT / "configs" / "tasks" / f"{args.task}.yaml")
    out = Path(args.out) if args.out else ROOT / "results" / "runs" / f"{args.task}.jsonl"
    writer = TrajectoryWriter(out)
    llm = make_llm(config)
    conditions = list(CONDITIONS) if args.all_conditions else [args.condition]

    if args.task == "custom_ds":
        episodes = generate_episodes(
            n_episodes=task_cfg.get("n_episodes", 10), seed=args.seed,
            n_steps=task_cfg.get("n_steps", 120),
            n_projects=task_cfg.get("n_projects", 6),
            n_queries=task_cfg.get("n_queries", 10),
            update_prob=task_cfg.get("update_prob", 0.25),
            distractor_prob=task_cfg.get("distractor_prob", 0.35),
        )[: args.limit]
        run = lambda cond, params: run_mode_b(  # noqa: E731
            episodes, cond, llm, config, writer, seed=args.seed, condition_params=params)
    else:
        data_file = ROOT / task_cfg["data_file"]
        if not data_file.exists():
            sys.exit(f"Missing {data_file}. See data/README.md for download steps.")
        items = load_longmemeval(
            data_file, cap=task_cfg.get("cap"), seed=args.seed,
            exclude_types=tuple(task_cfg.get("exclude_types", [])),
        )[: args.limit]
        run = lambda cond, params: run_mode_a(  # noqa: E731
            items, cond, llm, config, writer, seed=args.seed, condition_params=params)

    write_manifest(out.parent, {**config, "task": task_cfg, "cli": vars(args)})
    for cond in conditions:
        cond_cfg = load_yaml(ROOT / "configs" / "conditions" / f"{cond}.yaml")
        n = run(cond, cond_cfg.get("params", {}))
        print(f"[{args.task}] condition={cond} seed={args.seed}: "
              f"{n} new records -> {out}  (spent £{llm.cost.gbp_spent:.4f})")


if __name__ == "__main__":
    main()

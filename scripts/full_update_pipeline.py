"""End-to-end pipeline driver for the stroke triage notebook.

Runs:
1. Fairness threshold agent (metrics + fairness audit + DCA already baked in).
2. Advanced modeling experiments (regularized logistic, monotonic XGB, super learner).
3. Abstention zone analysis.

Usage (within the notebook or CLI):
    !python scripts/full_update_pipeline.py
"""

from __future__ import annotations

import traceback

from pathlib import Path

import sys

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(ROOT_DIR))

import fairness_threshold_agent  # type: ignore
import model_next_steps  # type: ignore
import abstention_analysis  # type: ignore


def main():
    errors = []
    steps = [
        ("Fairness threshold agent", fairness_threshold_agent.main),
        ("Advanced modeling experiments", model_next_steps.main),
        ("Abstention analysis", abstention_analysis.main),
    ]
    for label, func in steps:
        print(f"\n=== {label} ===")
        try:
            func()
        except Exception:  # noqa: BLE001
            print(f"[ERROR] {label} failed:\n{traceback.format_exc()}")
            errors.append(label)

    if errors:
        raise SystemExit(f"Pipeline completed with errors in steps: {', '.join(errors)}")
    print("\nPipeline completed successfully. Results stored in the 'results/' directory.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the existing fall cross-dataset evaluation against MobiAct v2.

This is a thin wrapper around ``run_fall_cross_dataset_eval.py`` that pins:
- ``--train-source combined``  (MobiFall + SisFall, the same training mix
  used for the locked headline numbers in Chapter 8)
- ``--eval-source mobiact_v2`` (external held-out corpus, never used in
  any training mix by design)
- ``--out-json results/validation/fall_cross_dataset_eval_mobiact_v2.json``

Resulting metrics are directly comparable to the existing
``fall_cross_dataset_eval.json`` (MobiFall / SisFall) numbers because all
three runs use the same threshold mode, window/step config, random state,
and meta-model hyperparameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_fall_cross_dataset_eval import main as run_cross_dataset_main  # noqa: E402


DEFAULT_OUT_JSON = "results/validation/fall_cross_dataset_eval_mobiact_v2.json"


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    def _ensure_arg(flag: str, value: str) -> None:
        if not any(arg == flag or arg.startswith(f"{flag}=") for arg in raw_args):
            raw_args.extend([flag, value])

    _ensure_arg("--train-source", "combined")
    _ensure_arg("--eval-source", "mobiact_v2")
    _ensure_arg("--out-json", DEFAULT_OUT_JSON)

    sys.argv = [sys.argv[0], *raw_args]
    return run_cross_dataset_main()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the existing HAR cross-dataset evaluation including MobiAct v2.

This is a thin wrapper around ``run_har_cross_dataset_eval.py`` that pins:
- ``--datasets uci_har,pamap2,mobiact_v2``  (adds MobiAct v2 to the existing
  UCIHAR/PAMAP2 cross-dataset block; WISDM is skipped here because WISDM has
  no subject IDs and the within-dataset path takes a different shape).
- ``--out-json results/validation/har_cross_dataset_eval_mobiact_v2.json``

The script trains a fresh RF per source-target pair (its existing behaviour),
so MobiAct v2 ends up in every direction: UCIHAR↔MOBIACT_V2 and
PAMAP2↔MOBIACT_V2. Read the resulting JSON's ``cross_dataset`` block to see
how the locked HAR meta-classifier-style transfer holds up against MobiAct v2.

Research-only output. Not wired into the dissertation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_har_cross_dataset_eval import main as run_har_cross_dataset_main  # noqa: E402


DEFAULT_OUT_JSON = "results/validation/har_cross_dataset_eval_mobiact_v2.json"
DEFAULT_DATASETS = "uci_har,pamap2,mobiact_v2"


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    def _ensure_arg(flag: str, value: str) -> None:
        if not any(arg == flag or arg.startswith(f"{flag}=") for arg in raw_args):
            raw_args.extend([flag, value])

    _ensure_arg("--datasets", DEFAULT_DATASETS)
    _ensure_arg("--out-json", DEFAULT_OUT_JSON)

    sys.argv = [sys.argv[0], *raw_args]
    return run_har_cross_dataset_main()


if __name__ == "__main__":
    raise SystemExit(main())

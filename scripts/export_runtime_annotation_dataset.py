from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.runtime_annotation_dataset import (
    RuntimeAnnotationConfig,
    load_runtime_annotation_dataset,
    save_runtime_annotation_dataset,
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an annotated runtime dataset from saved mobile session JSON files "
            "and export session/sample/segment/window CSVs."
        )
    )

    parser.add_argument(
        "--sessions-dir",
        type=Path,
        required=True,
        help="Directory containing saved session JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the exported dataset files will be written.",
    )
    parser.add_argument(
        "--session-glob",
        type=str,
        default="*.json",
        help="Glob pattern used to find session files inside the sessions directory.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=32,
        help="Minimum number of samples required for a session to be included.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Window size in samples for exported window labels.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=64,
        help="Step size in samples for exported window labels.",
    )
    parser.add_argument(
        "--require-segments",
        action="store_true",
        help="Only include sessions that contain explicit annotation segments.",
    )
    parser.add_argument(
        "--fallback-to-session-labels",
        action="store_true",
        help=(
            "If a session has no explicit segments, use the saved session-level "
            "activity/placement labels for all samples."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="APP_RUNTIME_ANNOTATED",
        help="Dataset name recorded in the exported summary config.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = RuntimeAnnotationConfig(
        session_glob=args.session_glob,
        min_samples=int(args.min_samples),
        window_size_samples=int(args.window_size),
        step_size_samples=int(args.step_size),
        require_segments=bool(args.require_segments),
        fallback_to_session_labels_for_unsegmented_sessions=bool(
            args.fallback_to_session_labels
        ),
        dataset_name=str(args.dataset_name),
    )

    dataset = load_runtime_annotation_dataset(
        args.sessions_dir,
        config=config,
    )
    outputs = save_runtime_annotation_dataset(
        dataset,
        output_dir=args.output_dir,
    )

    print("\nRuntime annotation dataset export complete.\n")
    print("Summary:")
    print(json.dumps(dataset.summary, indent=2))
    print("\nFiles written:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
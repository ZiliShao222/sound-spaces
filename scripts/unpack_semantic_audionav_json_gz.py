#!/usr/bin/env python3

"""Unpack .json.gz into .json without changing any content.

This script performs a lossless JSON.GZ -> JSON unpack:
- no key renaming
- no field filtering
- no schema normalization
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    name = path.name.lower()
    if name.endswith(".json.gz") or name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _default_output_path(input_path: Path) -> Path:
    name = input_path.name
    lower_name = name.lower()
    if lower_name.endswith(".json.gz"):
        base = name[: -len(".gz")]
        return input_path.with_name(base)
    if lower_name.endswith(".gz"):
        base = name[: -len(".gz")]
        return input_path.with_name(base)
    if lower_name.endswith(".json"):
        return input_path.with_name(f"{input_path.stem}.unpacked.json")
    return input_path.with_name(f"{name}.unpacked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Losslessly unpack .json.gz into .json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON.GZ path (.json.gz or .gz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default inferred from input)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise RuntimeError(f"Input JSON.GZ not found: {input_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _read_json(input_path)
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        ),
        encoding="utf-8",
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if isinstance(payload, dict):
        print(f"Root type: object, top-level keys: {len(payload)}")
    elif isinstance(payload, list):
        print(f"Root type: array, length: {len(payload)}")
    else:
        print(f"Root type: {type(payload).__name__}")


if __name__ == "__main__":
    main()

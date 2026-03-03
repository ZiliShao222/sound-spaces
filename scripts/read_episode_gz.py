#!/usr/bin/env python3

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and print a .json.gz episode file.")
    parser.add_argument("path", type=Path, help="Path to .json.gz file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    with gzip.open(args.path, "rt") as f:
        data = json.load(f)

    if args.pretty:
        print(json.dumps(data, indent=2, ensure_ascii=True))
    else:
        print(json.dumps(data, ensure_ascii=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check independence of the formal CLUTRR test and an external dev split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def row_key(row):
    selected = [row[index] for index in (2, 3, 5, 13) if index < len(row)]
    return hashlib.sha256(json.dumps(selected).encode("utf-8")).hexdigest()


def read_files(files):
    keys = []
    for path in sorted(files):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                if index == 0 and any(
                    value.lower() in {"id", "index", "story"} for value in row[:3]
                ):
                    continue
                if len(row) > 13:
                    keys.append(row_key(row))
    return keys


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    formal = Path(args.formal)
    development = Path(args.development)
    groups = {
        "formal_train": read_files(formal.glob("*_train.csv")),
        "formal_test": read_files(formal.glob("*_test.csv")),
        "development_test": read_files(development.glob("*_test.csv")),
    }
    report = {
        "groups": {
            name: {
                "rows": len(keys),
                "unique_rows": len(set(keys)),
                "duplicates": len(keys) - len(set(keys)),
            }
            for name, keys in groups.items()
        },
        "overlaps": {},
        "config_sha256": {
            "formal": file_hash(formal / "config.json"),
            "development": file_hash(development / "config.json"),
        },
    }
    for left, right in (
        ("formal_train", "formal_test"),
        ("formal_train", "development_test"),
        ("formal_test", "development_test"),
    ):
        report["overlaps"][f"{left}|{right}"] = len(
            set(groups[left]) & set(groups[right])
        )
    report["pass"] = not any(report["overlaps"].values())
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

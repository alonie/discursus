#!/usr/bin/env python3
"""Convert a local HLE parquet/CSV export into Discursus benchmark JSON format.

Usage:
  python scripts/build_hle_test_data.py \
    --input /path/to/hle.parquet \
    --output content/test_data_hle_100.json \
    --limit 100
"""

import argparse
import json
import os
import re
from typing import Any


def _first_present(record: dict, keys: list[str], default: Any = "") -> Any:
    for k in keys:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return default


def _extract_mcq_letter(answer: str) -> str:
    if not answer:
        return ""
    m = re.search(r"\b([A-J])\b", str(answer).strip().upper())
    return m.group(1) if m else ""

def _normalize_domain(raw: str) -> str:
    s = (raw or "hle").strip().lower()
    s = s.replace("&", "and")
    s = s.replace(" ", "_")
    return s


def _read_table(path: str) -> list[dict]:
    lower = path.lower()
    if lower.endswith(".parquet"):
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    if lower.endswith(".csv"):
        import pandas as pd

        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError("Unsupported input format. Use .parquet or .csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Local HLE parquet/csv export path")
    ap.add_argument("--output", required=True, help="Output benchmark JSON path")
    ap.add_argument("--limit", type=int, default=100, help="Number of cases to export")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    rows = _read_table(args.input)

    test_cases = []
    for i, row in enumerate(rows):
        q = str(
            _first_present(
                row,
                [
                    "question",
                    "question_text",
                    "prompt",
                    "problem",
                    "query",
                    "item",
                ],
                "",
            )
        ).strip()
        if not q:
            continue

        domain = str(_first_present(row, ["raw_subject", "domain", "category", "subject", "field"], "hle")).strip() or "hle"
        answer_raw = str(_first_present(row, ["answer", "final_answer", "ground_truth", "label"], "")).strip()
        answer_type = str(_first_present(row, ["answer_type"], "")).strip()
        answer_letter = _extract_mcq_letter(answer_raw)

        case = {
            "id": f"hle_case_{len(test_cases):03d}",
            "domain": _normalize_domain(domain),
            "complexity": "hle",
            "title": f"HLE Case {len(test_cases) + 1}",
            "prompt": f"Question: {q}",
            "metadata": {
                "source_benchmark": "Humanity's Last Exam (HLE)",
                "subset": "local_export",
                "row_index": i,
                "category": str(_first_present(row, ["category"], "")),
                "raw_subject": str(_first_present(row, ["raw_subject"], "")),
                "answer_type": answer_type,
            },
        }

        if answer_type.lower() == "exactmatch" and answer_raw:
            case["metadata"]["expected_answer_text"] = answer_raw
            case["metadata"]["scoring"] = "exact_match_text"
        elif answer_letter:
            case["metadata"]["correct_answer"] = answer_letter
            case["metadata"]["scoring"] = "exact_match_letter"
        elif answer_raw:
            case["metadata"]["expected_answer_text"] = answer_raw
            case["metadata"]["scoring"] = "manual_or_custom"

        test_cases.append(case)
        if len(test_cases) >= args.limit:
            break

    payload = {
        "benchmark": "HLE local export",
        "notes": "Converted from local HLE export for Discursus benchmarking portal.",
        "test_cases": test_cases,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output} with {len(test_cases)} test cases")


if __name__ == "__main__":
    main()

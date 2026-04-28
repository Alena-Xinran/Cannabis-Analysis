import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .config import ANNOTATION_DIR, RESEARCH_DATASET_CSV


def allocate_counts(group_sizes: pd.Series, target_rows: int) -> dict[str, int]:
    total = int(group_sizes.sum())
    if total == 0:
        return {}

    allocations = {}
    for group_name, size in group_sizes.items():
        n = max(1, int(math.floor(target_rows * (size / total))))
        allocations[group_name] = min(n, int(size))

    current_total = sum(allocations.values())
    if current_total < target_rows:
        remaining = target_rows - current_total
        order = sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)
        idx = 0
        while remaining > 0 and order:
            group_name, size = order[idx % len(order)]
            if allocations[group_name] < int(size):
                allocations[group_name] += 1
                remaining -= 1
            idx += 1
            if idx > len(order) * 10 and all(allocations[g] >= int(s) for g, s in order):
                break

    return allocations


def build_second_batch(
    dataset_csv: Path,
    processed_csv: Path,
    output_csv: Path,
    target_total: int,
    random_seed: int = 42,
) -> dict:
    full_df = pd.read_csv(dataset_csv, usecols=["text_id", "source", "date_utc", "text"])
    full_df = full_df[full_df["source"] == "comment"].copy()
    full_df = full_df[full_df["text"].fillna("").ne("")].copy()
    full_df["text_id"] = full_df["text_id"].astype(str)

    if processed_csv.exists():
        processed_df = pd.read_csv(processed_csv, usecols=["text_id"])
        processed_ids = set(processed_df["text_id"].astype(str))
    else:
        processed_ids = set()

    processed_count = len(processed_ids)
    target_rows = max(0, target_total - processed_count)

    remaining_df = full_df[~full_df["text_id"].isin(processed_ids)].copy()
    remaining_df["date_utc"] = pd.to_datetime(remaining_df["date_utc"], utc=True, errors="coerce")
    remaining_df = remaining_df.dropna(subset=["date_utc"]).copy()
    remaining_df["year_month"] = remaining_df["date_utc"].dt.to_period("M").astype(str)

    if target_rows <= 0 or remaining_df.empty:
        sample_df = remaining_df.head(0).copy()
    else:
        group_sizes = remaining_df.groupby("year_month").size()
        allocations = allocate_counts(group_sizes, target_rows)
        samples = []
        for year_month, group in remaining_df.groupby("year_month"):
            n = allocations.get(year_month, 0)
            if n <= 0:
                continue
            samples.append(group.sample(n=min(n, len(group)), random_state=random_seed))
        sample_df = pd.concat(samples, ignore_index=True) if samples else remaining_df.head(0).copy()
        if len(sample_df) > target_rows:
            sample_df = sample_df.sample(n=target_rows, random_state=random_seed).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_df.drop(columns=["year_month"], errors="ignore").to_csv(output_csv, index=False)

    summary = {
        "processed_count": processed_count,
        "target_total": target_total,
        "batch2_rows": int(len(sample_df)),
        "remaining_comments": int(len(remaining_df)),
        "year_month_counts": sample_df["date_utc"].pipe(pd.to_datetime, utc=True, errors="coerce").dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create batch 2 by year-month stratified sampling from remaining comments.")
    parser.add_argument("--dataset-csv", type=Path, default=RESEARCH_DATASET_CSV)
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=ANNOTATION_DIR / "pair_level_dataset_gpt41mini_incremental.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ANNOTATION_DIR / "batch2_year_month_sample.csv",
    )
    parser.add_argument("--target-total", type=int, default=10000)
    args = parser.parse_args()

    summary = build_second_batch(
        dataset_csv=args.dataset_csv,
        processed_csv=args.processed_csv,
        output_csv=args.output_csv,
        target_total=args.target_total,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

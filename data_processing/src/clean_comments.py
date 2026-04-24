import argparse
from pathlib import Path

from .config import FINAL_DIR, RAW_COMMENTS_ZIP, RAW_POSTS_ZIP, RESEARCH_DATASET_CSV
from .preprocess import build_clean_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned dataset used for pair extraction.")
    parser.add_argument("--posts-zip", type=Path, default=RAW_POSTS_ZIP)
    parser.add_argument("--comments-zip", type=Path, default=RAW_COMMENTS_ZIP)
    parser.add_argument("--output-csv", type=Path, default=RESEARCH_DATASET_CSV)
    args = parser.parse_args()

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    df = build_clean_dataset(args.posts_zip, args.comments_zip)
    df.to_csv(args.output_csv, index=False)
    print(args.output_csv)
    print(len(df))


if __name__ == "__main__":
    main()

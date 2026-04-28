from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FINAL_DIR = DATA_DIR / "processed" / "final"
ANNOTATION_DIR = DATA_DIR / "processed" / "annotation"

RAW_POSTS_ZIP = RAW_DIR / "reddit_posts.zip"
RAW_COMMENTS_ZIP = RAW_DIR / "reddit_comments.zip"

RESEARCH_DATASET_CSV = FINAL_DIR / "research_dataset.csv"
PAIR_LEVEL_DATASET_CSV = ANNOTATION_DIR / "pair_level_dataset_gpt41mini.csv"

ALLOWED_PRODUCTS = ["flower", "oil", "gummies", "vape", "topical"]
ALLOWED_SENTIMENTS = ["positive", "negative", "neutral"]

MIN_TOKEN_LENGTH = 3
MAX_TEXT_LENGTH = 5000

REDDIT_BOILERPLATE = {
    "",
    "[deleted]",
    "[removed]",
    "deleted",
    "removed",
    "n/a",
}

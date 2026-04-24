import html
import re
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import MAX_TEXT_LENGTH, MIN_TOKEN_LENGTH, REDDIT_BOILERPLATE


URL_RE = re.compile(r"https?://\S+")
MARKDOWN_RE = re.compile(r"[*_~`]+")
MENTION_RE = re.compile(r"\b[ru]/\w+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    if pd.isna(raw):
        return ""
    text = str(raw).strip()
    if text.lower() in REDDIT_BOILERPLATE:
        return ""
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = MARKDOWN_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text[:MAX_TEXT_LENGTH]


def token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def read_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        filename = zf.namelist()[0]
        with zf.open(filename) as fp:
            return pd.read_csv(fp)


def load_posts(posts_zip: Path) -> pd.DataFrame:
    df = read_csv_from_zip(posts_zip).copy()
    df["text"] = df["full_text"].apply(clean_text)
    df["source"] = "post"
    df["text_id"] = df["id"].astype(str)
    return df


def load_comments(comments_zip: Path) -> pd.DataFrame:
    df = read_csv_from_zip(comments_zip).copy()
    df["text"] = df["body"].apply(clean_text)
    df["source"] = "comment"
    df["text_id"] = df["id"].astype(str)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="text_id", keep="first")
    df = df.drop_duplicates(subset="text", keep="first")
    return df


def filter_short_records(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["text"].apply(token_count) >= MIN_TOKEN_LENGTH].copy()


def base_columns(df: pd.DataFrame, text_field_fallback: str) -> pd.DataFrame:
    wanted = [
        "text_id",
        "id",
        "source",
        "subreddit",
        "author",
        "score",
        "created_utc",
        "date_utc",
        "year",
        "month",
        "searched_subreddit",
        "searched_keyword",
        "text",
    ]
    for col in wanted:
        if col not in df.columns:
            df[col] = None
    if text_field_fallback not in df.columns:
        df[text_field_fallback] = None
    return df[wanted].copy()


def unify_raw_tables(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    posts = base_columns(posts, "full_text")
    comments = base_columns(comments, "body")
    df = pd.concat([posts, comments], ignore_index=True)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")
    df["text_length_chars"] = df["text"].str.len()
    df["text_length_tokens"] = df["text"].apply(token_count)
    df["is_deleted_or_empty"] = df["text"].eq("")
    df = df.sort_values(["date_utc", "text_id"], na_position="last").reset_index(drop=True)
    return df


def build_clean_dataset(posts_zip: Path, comments_zip: Path) -> pd.DataFrame:
    posts = filter_short_records(deduplicate(load_posts(posts_zip)))
    comments = filter_short_records(deduplicate(load_comments(comments_zip)))
    return unify_raw_tables(posts, comments)


def summarize_counts(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "posts": int((df["source"] == "post").sum()),
        "comments": int((df["source"] == "comment").sum()),
        "subreddits": int(df["subreddit"].nunique(dropna=True)),
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
    }

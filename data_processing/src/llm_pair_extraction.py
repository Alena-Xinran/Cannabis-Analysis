import argparse
import csv
import json
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from .config import ALLOWED_PRODUCTS, ALLOWED_SENTIMENTS, RESEARCH_DATASET_CSV


SYSTEM_PROMPT = """You are an information extraction annotator for a cannabis Reddit dataset.

Extract all valid (product, sentiment) pairs from one Reddit comment.

Allowed product labels:
- flower
- oil
- gummies
- vape
- topical

Allowed sentiment labels:
- positive
- negative
- neutral

Rules:
1. A single comment may contain multiple pairs.
2. Only extract a pair if the comment expresses an opinion, experience, preference, concern, or evaluation about that product.
3. If a product is mentioned with no clear sentiment, do not extract a pair.
4. If the same product appears multiple times with the same overall sentiment, return only one pair for that product.
5. If the same product has mixed sentiment in the same comment, return sentiment = neutral.
6. Use these mappings:
   - tincture, CBD oil, THC oil, drops, droppers, sublingual oil, oral oil -> oil
   - vape oil, vape pen, cart, cartridge, disposable vape, dab pen, vape juice -> vape
   - gummy, gummies, edible gummy -> gummies
   - bud, joint, blunt, preroll, nug, flower -> flower
   - cream, salve, balm, lotion, patch, topical -> topical
7. Do not invent products that are not supported by the text.
8. Return strict JSON only.
9. If no valid pair exists, return {"pairs": []}.

Return format:
{
  "pairs": [
    {"product": "gummies", "sentiment": "positive"},
    {"product": "vape", "sentiment": "negative"}
  ]
}"""


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def strip_json_block(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


def parse_pairs_response(text: str) -> list[dict]:
    raw = strip_json_block(text)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain JSON.")
    data = json.loads(raw[start : end + 1])
    pairs = data.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("`pairs` must be a list.")

    cleaned = []
    seen = set()
    for item in pairs:
        if not isinstance(item, dict):
            continue
        product = str(item.get("product", "")).strip().lower()
        sentiment = str(item.get("sentiment", "")).strip().lower()
        if product not in ALLOWED_PRODUCTS:
            continue
        if sentiment not in ALLOWED_SENTIMENTS:
            continue
        key = (product, sentiment)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"product": product, "sentiment": sentiment})
    return cleaned


def build_user_prompt(text_id: str, text: str) -> str:
    return (
        "Extract all (product, sentiment) pairs from this Reddit comment.\n\n"
        f"Comment ID: {text_id}\n\n"
        f"Text:\n{text}"
    )


def extract_pairs(client: OpenAI, model: str, text_id: str, text: str) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text_id=text_id, text=text)},
        ],
    )
    content = response.choices[0].message.content or ""
    return parse_pairs_response(content)


def prepare_input(input_csv: Path, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(input_csv, usecols=["text_id", "source", "date_utc", "text"])
    df = df[df["source"] == "comment"].copy()
    df = df[df["text"].fillna("").ne("")].copy()
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return df


def run_pair_extraction(
    input_csv: Path,
    output_csv: Path,
    model: str,
    max_rows: int | None,
    sleep_seconds: float,
) -> None:
    client = get_client()
    df = prepare_input(input_csv=input_csv, max_rows=max_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "text_id",
        "product",
        "sentiment",
        "source",
        "date_utc",
        "response_id",
        "extraction_error",
    ]

    processed_ids = set()
    if output_csv.exists():
        existing = pd.read_csv(output_csv, usecols=["text_id"])
        processed_ids = set(existing["text_id"].astype(str))

    if not output_csv.exists():
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    remaining = df[~df["text_id"].astype(str).isin(processed_ids)].copy()

    with output_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Extracting pairs"):
            text_id = str(row["text_id"])
            response_id = ""
            output_rows = []
            wrote_any = False
            wrote_no_pair = False
            error_message = ""

            try:
                text = str(row["text"])
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(text_id=text_id, text=text)},
                    ],
                )
                response_id = getattr(response, "id", "")
                content = response.choices[0].message.content or ""
                pairs = parse_pairs_response(content)

                if pairs:
                    for pair in pairs:
                        output_rows.append(
                            {
                                "text_id": text_id,
                                "product": pair["product"],
                                "sentiment": pair["sentiment"],
                                "source": row["source"],
                                "date_utc": row["date_utc"],
                                "response_id": response_id,
                                "extraction_error": "",
                            }
                        )
                    writer.writerows(output_rows)
                    wrote_any = True
                else:
                    writer.writerow(
                        {
                            "text_id": text_id,
                            "product": "",
                            "sentiment": "",
                            "source": row["source"],
                            "date_utc": row["date_utc"],
                            "response_id": response_id,
                            "extraction_error": "",
                        }
                    )
                    wrote_no_pair = True
            except Exception as exc:
                error_message = str(exc)
                writer.writerow(
                    {
                        "text_id": text_id,
                        "product": "",
                        "sentiment": "",
                        "source": row["source"],
                        "date_utc": row["date_utc"],
                        "response_id": response_id,
                        "extraction_error": error_message,
                    }
                )

            f.flush()
            os.fsync(f.fileno())

            if sleep_seconds:
                time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pair-level labels from 10,000 cleaned comments.")
    parser.add_argument("--input-csv", type=Path, default=RESEARCH_DATASET_CSV)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/annotation/pair_level_dataset_gpt41mini_incremental.csv"),
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    run_pair_extraction(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model=args.model,
        max_rows=args.max_rows,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()

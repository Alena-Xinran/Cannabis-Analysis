# Pair-Level Cannabis Reddit Study

This project focuses on one pipeline:

**clean Reddit comments -> use `GPT-4.1-mini` to extract multiple `(product, sentiment)` pairs -> analyze the pair-level dataset**

## Keep

- `src/clean_comments.py`
  Builds the cleaned dataset.
- `src/preprocess.py`
  Core cleaning logic.
- `src/llm_pair_extraction.py`
  Extracts pair-level labels from cleaned comments.
- `data/processed/final/research_dataset.csv`
  Cleaned input dataset.
- `data/processed/annotation/pair_level_dataset_gpt41mini.csv`
  Final pair-level dataset.
- `Pair_Level_Paper.ipynb`
  Paper-style notebook for analysis and modeling.

## Final pair-level format

```text
text_id | product | sentiment| date_utc
```

One `text_id` may appear in multiple rows.

## Run

Build the cleaned dataset:

```bash
python -m src.clean_comments
```

Run pair extraction on 10,000 cleaned comments:

```bash
export OPENAI_API_KEY='YOUR_KEY'
export OPENAI_BASE_URL='https://api.chatanywhere.org/v1'
bash run.sh
```

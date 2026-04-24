# Cannabis Product Sentiment Analysis on Reddit

This repository contains the data collection and data processing work for a temporal sentiment analysis project on cannabis-derived products discussed on Reddit.

## Project Overview

The project uses Reddit posts and comments from cannabis-related communities, including r/CBD, r/cannabis, r/trees, r/weed, and r/Marijuana. The raw data cover discussions from 2022 to 2025 and support analysis of public discussion and sentiment around cannabis-derived products such as flower, oils, tinctures, gummies, edibles, vapes, and topicals.

The processing pipeline cleans Reddit text, samples records across the available time range, uses `GPT-4.1-mini` to extract product-specific sentiment pairs, and analyzes product-level sentiment trends over time.

## Repository Structure

```text
data_collection/
  Raw Reddit dataset files and data collection notebook.

data_processing/
  Cleaning scripts, GPT pair extraction code, processed datasets, final pair-level data, and analysis notebook.
```

## Data Collection

The raw datasets consist of Reddit posts and comments collected from cannabis-related communities. These files are stored under:

```text
data_collection/data/raw/
```

The collection notebook is:

```text
data_collection/download_reddit_dataset.ipynb
```

## Data Processing Pipeline

The processing workflow is:

```text
clean Reddit comments -> use GPT-4.1-mini to extract multiple (product, sentiment) pairs -> analyze the pair-level dataset
```

Important files:

```text
data_processing/src/clean_comments.py
data_processing/src/preprocess.py
data_processing/src/llm_pair_extraction.py
data_processing/src/sample_batch2_year_month.py
data_processing/Pair_Level_Paper.ipynb
data_processing/data_description.md
```

The GPT annotation input was built as a 10,000-row time-aware sample from the cleaned Reddit data. Records were sorted by time and sampled across the available date range, rather than using the first 10,000 rows. This keeps multiple time periods represented for temporal analysis.

## Final Pair-Level Dataset

The final cleaned pair-level dataset is:

```text
data_processing/data/final/pair_level_gpt41mini_clean.csv
```

Final schema:

```text
text_id | product | sentiment | date_utc
```

Allowed product labels:

```text
flower, oil, gummies, vape, topical
```

Allowed sentiment labels:

```text
positive, negative, neutral
```

One `text_id` may appear in multiple rows because a single Reddit text can mention multiple products or contain multiple product-specific opinions.

## Run

Install requirements:

```bash
pip install -r data_processing/requirements.txt
```

Build the cleaned dataset:

```bash
cd data_processing
python -m src.clean_comments
```

Run GPT pair extraction:

```bash
cd data_processing
export OPENAI_API_KEY='YOUR_KEY'
export OPENAI_BASE_URL='https://api.chatanywhere.org/v1'
bash run.sh
```

Run the analysis notebook:

```text
data_processing/Pair_Level_Paper.ipynb
```

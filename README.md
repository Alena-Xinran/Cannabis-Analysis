# Cannabis Reddit Sentiment Analysis

This project analyzes Reddit posts and comments about cannabis-related topics, focusing on discussion trends and sentiment changes for different cannabis-derived products from 2022 to 2024.

The main workflow includes:

- Collecting raw Reddit data
- Cleaning text and building sampled datasets
- Using GPT-4.1-mini to extract pair-level `(product, sentiment)` labels
- Running statistical analysis and modeling on the final dataset

## Project Structure

```text
data_collection/    Raw Reddit data and download notebook
data_processing/    Data cleaning, sampling, LLM extraction, and analysis outputs
Modeling/           Product label prediction and sentiment classification notebooks
Result/             Final result and ROC-related notebook
```

## Main Data Files

```text
data_processing/data/processed/cleaned_dataset.csv
data_processing/data/processed/year_month_sample.csv
data_processing/data/final/pair_level_gpt41mini_clean.csv
data_processing/data/final/analysis_outputs/
```

The final pair-level dataset has the following format:

```text
text_id | product | sentiment | date_utc
```

The `product` labels include `flower`, `oil`, `gummies`, `vape`, and `topical`. The `sentiment` labels include `positive`, `negative`, and `neutral`.

## Run

Install dependencies:

```bash
pip install -r data_processing/requirements.txt
```

Clean the data:

```bash
cd data_processing
python -m src.clean_comments
```

Run LLM pair extraction:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="YOUR_BASE_URL"
bash run.sh
```

For more details about the dataset, see:

```text
data_processing/data_description.md
```

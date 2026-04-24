# Cannabis Analysis

This repository contains two parts of the project:

```text
data_collection/   Raw Reddit data files and collection notebook
data_processing/   Cleaning, pair extraction, final datasets, and analysis notebook
```

For dataset details, sampling design, label definitions, and final pair-level schema, see:

```text
data_processing/data_description.md
```

## Main Outputs

```text
data_processing/data/final/pair_level_gpt41mini_clean.csv
data_processing/Pair_Level_Paper.ipynb
```

## Run

```bash
pip install -r data_processing/requirements.txt
cd data_processing
python -m src.clean_comments
bash run.sh
```

`run.sh` requires an OpenAI-compatible API key in the environment.

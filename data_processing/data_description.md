# Data Description

The raw datasets contain Reddit posts and comments from cannabis-related communities, including r/CBD, r/cannabis, r/trees, r/weed, and r/Marijuana. The data cover discussions from 2022 to 2025 and support temporal analysis of public discussion and sentiment around cannabis-derived products such as flower, oils, tinctures, gummies, edibles, vapes, and topicals.

This project first cleaned a Reddit dataset by standardizing the text, removing URLs and markdown artifacts, stripping user and subreddit mentions, removing duplicate entries, and filtering out very short or low-information comments. It then used `GPT-4.1-mini` to extract multiple `(product, sentiment)` pairs from individual comments. Instead of assigning one product label and one sentiment label to an entire comment, the project treats a single comment as potentially containing multiple product-specific opinions. The final pair-level dataset stores one row per extracted pair.

The GPT annotation input was built as a 10,000-row time-aware sample from the cleaned Reddit data. The records were sorted by time and sampled to cover the available time range, rather than simply taking the first 10,000 rows. This keeps observations from different periods in the final pair-level dataset and supports product-specific temporal analysis.

The target output format is:

`text_id | product | sentiment | date_utc`

The allowed product labels are `flower`, `oil`, `gummies`, `vape`, and `topical`, and the allowed sentiment labels are `positive`, `negative`, and `neutral`. Because one comment may discuss multiple products, the same `text_id` can appear in multiple rows. The pair-level dataset is then used for downstream tasks such as pair-level classification and product-specific temporal analysis.

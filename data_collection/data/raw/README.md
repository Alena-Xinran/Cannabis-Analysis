## Raw Datasets

This project uses two raw Reddit datasets collected from cannabis-related online communities for the period January 1, 2022 through January 1, 2025. The purpose of these datasets is to support temporal sentiment analysis of consumer discussions surrounding cannabis-derived products, including oils, tinctures, gummies, edibles, vapes, and topicals.

The raw data were collected from the following subreddits:

•	r/CBD

•	r/cannabis

•	r/trees

•	r/weed

•	r/Marijuana

Data collection was performed using the PullPush Reddit search API, which provides access to historical Reddit content. We queried both the submission endpoint for posts and the comment endpoint for comments. To identify relevant discussions, we searched using cannabis- and product-related keywords such as CBD, edible, gummies, vape, oil, tincture, and topical. Queries were run in monthly time intervals to collect historical data across the full study period. The resulting raw datasets were saved into two separate files:

•	reddit_posts.csv

•	reddit_comments.csv

### Dataset 1: reddit_posts.csv

This file contains Reddit posts collected from the selected subreddits.

Column descriptions:

•	id: Unique Reddit identifier for each post.

•	subreddit: Name of the subreddit where the post was published.

•	title: Title of the Reddit post.

•	selftext: Main body text of the post. This field may be empty if the post only contains a title or external link.

•	author: Username of the post author.

•	created_utc: Unix timestamp representing the post creation time in Coordinated Universal Time.

•	score: Net Reddit score of the post, based on voting activity.

•	num_comments: Number of comments associated with the post at the time of collection.

•	url: URL linked to the post or the Reddit post link itself.

•	searched_subreddit: Subreddit specified in the API query used to retrieve the post.

•	searched_keyword: Keyword used in the API query that matched the post.

•	source_type: Data source type; for this file, the value is submission.

•	date_utc: Human-readable UTC date and time converted from created_utc.

•	year: Year extracted from date_utc.

•	month: Month extracted from date_utc.

•	full_text: Combined text field created by concatenating title and selftext for downstream text analysis.

### Dataset 2: reddit_comments.csv

This file contains Reddit comments collected from the selected subreddits.

Column descriptions:

•	id: Unique Reddit identifier for each comment.

•	subreddit: Name of the subreddit where the comment was posted.

•	author: Username of the comment author.

•	body: Full text content of the comment.

•	created_utc: Unix timestamp representing the comment creation time in Coordinated Universal Time.

•	score: Net Reddit score of the comment, based on voting activity.

•	link_id: Identifier of the Reddit submission associated with the comment.

•	parent_id: Identifier of the parent object of the comment, which may be either the original post or another comment.

•	searched_subreddit: Subreddit specified in the API query used to retrieve the comment.

•	searched_keyword: Keyword used in the API query that matched the comment.

•	source_type: Data source type; for this file, the value is comment.

•	date_utc: Human-readable UTC date and time converted from created_utc.

•	year: Year extracted from date_utc.

•	month: Month extracted from date_utc.

## Notes

These datasets are considered raw data because they were collected directly from the API before final filtering, labeling, and modeling. Duplicate records were removed based on Reddit IDs, and basic text cleaning was performed to standardize formatting. These raw files serve as the starting point for subsequent preprocessing, sentiment classification, and temporal analysis in this class project.

"""Clean the collected posts and balance the three labels.

Input:  data/posts_raw.csv (the collected posts, not versioned) and data/claims.csv
Output: data/posts_clean.csv

Each post inherits the verdict of the fact-checked claim whose keywords
retrieved it: the label describes the claim, not what the post says.
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
MIN_WORDS = 5
SEED = 42


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # links
    text = re.sub(r"u/\w+|r/\w+", "", text)  # Reddit user and subreddit mentions
    text = re.sub(r"\[removed\]|\[deleted\]", "", text)
    text = re.sub(r"[^\w\s'\-àâçéèêëîïôùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŸŒÆ.,!?;:]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def n_words(text: str) -> int:
    return len(str(text).split())


def main() -> None:
    posts = pd.read_csv(DATA / "posts_raw.csv")
    claims = pd.read_csv(DATA / "claims.csv")
    print(f"Collected posts: {len(posts)}")

    posts = posts.drop_duplicates("post_id").drop_duplicates("text")
    posts = posts[posts["text"].map(n_words) >= MIN_WORDS]
    posts = posts[posts["subreddit"] != "ActualiteFrance"]  # a single post came from this subreddit
    posts = posts.assign(text_clean=posts["text"].map(clean_text))
    posts = posts[posts["text_clean"].map(n_words) >= MIN_WORDS]

    posts = posts.merge(claims[["claim_id", "verdict", "theme"]], on="claim_id", how="left")
    posts = posts.rename(columns={"verdict": "label"})
    print(f"After cleaning: {len(posts)} posts, labels {posts['label'].value_counts().to_dict()}")

    # Undersample every label to the size of the smallest one.
    n = posts["label"].value_counts().min()
    balanced = pd.concat(
        [posts[posts["label"] == label].sample(n=n, random_state=SEED) for label in posts["label"].unique()]
    ).sample(frac=1, random_state=SEED)

    cols = ["post_id", "claim_id", "label", "theme", "subreddit", "created_utc", "text_clean"]
    balanced[cols].to_csv(DATA / "posts_clean.csv", index=False)
    print(f"Balanced: {len(balanced)} posts ({n} per label), "
          f"{balanced['claim_id'].nunique()} claims -> data/posts_clean.csv")


if __name__ == "__main__":
    main()

"""Collect the Reddit posts with the official Reddit API (PRAW).

Two modes:
  python src/collect.py --ids      re-download the exact posts of this project (data/post_ids.csv)
  python src/collect.py --search   run the keyword search of each claim again (results change over time)

Output: data/posts_raw.csv (post_id, claim_id, subreddit, created_utc, text). It is
not versioned: post text stays on Reddit, and deleted posts are not re-downloaded.

Credentials come from environment variables. Since November 2025, Reddit grants new API
credentials only on request (Responsible Builder Policy):
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT (e.g. "disinfo-study by u/<username>")
PRAW follows Reddit's rate limits on its own.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
SUBREDDITS = ["france", "francophonie", "ActualiteFrance", "politique"]
REMOVED = {"[removed]", "[deleted]"}


def reddit_client():
    import praw

    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def to_row(submission, claim_id: int) -> dict:
    text = submission.title
    if submission.selftext:
        text += " " + submission.selftext
    return {
        "post_id": submission.id,
        "claim_id": claim_id,
        "subreddit": submission.subreddit.display_name,
        "created_utc": int(submission.created_utc),
        "text": text.strip(),
    }


def search(reddit, claims: pd.DataFrame, limit: int) -> list[dict]:
    """Search each subreddit with each claim's keywords; a post keeps its first claim."""
    rows, seen = [], set()
    for claim in claims.itertuples():
        found = 0
        for name in SUBREDDITS:
            for post in reddit.subreddit(name).search(
                claim.search_keywords, sort="relevance", time_filter="all", limit=limit
            ):
                if post.id in seen or post.selftext in REMOVED:
                    continue
                seen.add(post.id)
                rows.append(to_row(post, claim.claim_id))
                found += 1
        print(f"claim {claim.claim_id:>2} ({claim.verdict}): {found} posts")
    return rows


def rehydrate(reddit, ids: pd.DataFrame) -> list[dict]:
    """Download the listed posts again; PRAW requests them 100 at a time."""
    claim_of = dict(zip(ids["post_id"], ids["claim_id"]))
    posts = reddit.info(fullnames=[f"t3_{post_id}" for post_id in ids["post_id"]])
    return [to_row(post, claim_of[post.id]) for post in posts if post.selftext not in REMOVED]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ids", action="store_true", help="re-download the posts listed in data/post_ids.csv")
    mode.add_argument("--search", action="store_true", help="search Reddit again with each claim's keywords")
    parser.add_argument("--limit", type=int, default=100, help="results per claim and subreddit (--search)")
    args = parser.parse_args()

    reddit = reddit_client()
    if args.ids:
        ids = pd.read_csv(DATA / "post_ids.csv")
        rows = rehydrate(reddit, ids)
        print(f"{len(rows)} of {len(ids)} posts are still online")
    else:
        rows = search(reddit, pd.read_csv(DATA / "claims.csv"), args.limit)
    pd.DataFrame(rows).to_csv(DATA / "posts_raw.csv", index=False)
    print(f"{len(rows)} posts -> data/posts_raw.csv")


if __name__ == "__main__":
    main()

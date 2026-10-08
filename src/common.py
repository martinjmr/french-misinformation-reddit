"""Shared paths, text cleaning, data loading and cross-validation folds."""
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
SEEDS = (0, 1, 2)
N_FOLDS = 5
LEAD_WORDS = 60  # the title and the opening of the post

# Common French function words, removed before comparing a claim with a post.
STOPWORDS = set("""
a afin ai aie aient ainsi alors as au aucun aujourd aupres aussi autre autres aux avaient avais avait avant avec
avez avoir avons ayant c ca car ce ceci cela celle celles celui cependant certains ces cet cette ceux chez ci comme
comment contre d dans de depuis des deux devrait dire dit dois doit donc dont du elle elles en encore entre es est
et etaient etais etait ete etre eu eux fait faire fois font hors i il ils j je jusqu l la le les leur leurs lors
lui m ma mais me meme memes mes moi moins mon n ne ni non nos notre nous on ont ou par parce pas peu peut peuvent
plus pour pourquoi qu quand que quel quelle quelles quels qui quoi s sa sans se selon ses si sien soit son sont
sous suis sur t ta tant te tes toi ton tous tout toute toutes tres tu un une unes uns va vers veut via voici voila
vont vos votre vous y
""".split())


def clean_text(text: str) -> str:
    """Lowercase, remove links, user and subreddit mentions, and stray symbols."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"u/\w+|r/\w+", "", text)
    text = re.sub(r"\[removed\]|\[deleted\]|&amp;#x200b;|&amp;|&gt;|&lt;", " ", text)
    text = re.sub(r"[^\w\s'\-àâçéèêëîïôùûüÿœæ.,!?;:]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fold_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def content_words(text: str) -> list[str]:
    """Accent-free words of three letters or more, without function words."""
    words = re.findall(r"[a-z0-9]+", fold_accents(str(text).lower()))
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def lead(text: str, n: int = LEAD_WORDS) -> str:
    return " ".join(str(text).split()[:n])


def load_posts() -> pd.DataFrame:
    """All collected posts with their claim, the claim's verdict and the stance label.

    Needs data/posts_raw.csv (the collected posts, not versioned).
    """
    posts = pd.read_csv(DATA / "posts_raw.csv")
    claims = pd.read_csv(DATA / "claims.csv")
    labels = pd.read_csv(DATA / "stance_labels.csv")
    df = (posts.merge(labels[["post_id", "stance"]], on="post_id", how="inner")
          .merge(claims[["claim_id", "claim", "verdict", "theme", "search_keywords"]], on="claim_id"))
    assert len(df) == len(posts) == len(labels), "every collected post needs a stance label"
    df["text_clean"] = df["text"].map(clean_text)
    df["lead_clean"] = df["text_clean"].map(lead)
    df["relevant"] = (df["stance"] != "off_topic").astype(int)
    return df.sort_values("post_id").reset_index(drop=True)


def claim_folds(df: pd.DataFrame, y: pd.Series, seed: int):
    """5 folds in which all posts of a claim share the same fold."""
    return StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(df, y, groups=df["claim_id"])

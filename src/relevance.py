"""Does a post actually discuss the fact-checked claim that retrieved it?

Input:  data/posts_raw.csv (not versioned), data/claims.csv, data/stance_labels.csv,
        and, when present, reports/transformers/relevance_scores.csv (written by the Colab notebook)
Output: data/folds.csv, reports/relevance/{results.json, results.md, errors_seed0.csv, ap_{light,dark}.png}

A post is relevant when its stance label is supports, refutes or discusses.
Every model gives each (claim, post) pair a score; posts are ranked by score.
Evaluation: 5 folds grouped by claim (a test claim never appears in training), 3 seeds.
- average precision (AP): area under the precision-recall curve; a random ranking scores the share of relevant posts
- reading load: share of the fold's posts to read, best-ranked first, to find 80% of its relevant posts
"""
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import REPORTS, DATA, SEEDS, claim_folds, content_words, lead, load_posts
from plots import hbar

OUT = REPORTS / "relevance"
TRANSFORMER_SCORES = REPORTS / "transformers" / "relevance_scores.csv"
TARGET_RECALL = 0.8


# ---------------------------------------------------------------- features
def stem(word: str) -> str:
    return word[:5]  # crude French stemming: "vaccins", "vaccination" -> "vacci"


def keyword_share(keywords: str, text: str) -> float:
    """Share of the claim's search keywords found in the text (compared on 5-letter stems)."""
    kws = {stem(w) for w in content_words(keywords)}
    words = {stem(w) for w in content_words(text)}
    return len(kws & words) / max(len(kws), 1)


def tfidf_cosines(df: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between the claim and the post, on the whole post and on its lead.

    The vocabulary and IDF weights are fitted on post texts only, without labels.
    """
    vec = TfidfVectorizer(analyzer=lambda t: [stem(w) for w in content_words(t)], sublinear_tf=True, min_df=2)
    vec.fit(df["text_clean"])
    claims = vec.transform(df["claim"])
    full = vec.transform(df["text_clean"])
    leads = vec.transform(df["lead_clean"])
    return pd.DataFrame({
        "cos_full": np.asarray(claims.multiply(full).sum(axis=1)).ravel(),
        "cos_lead": np.asarray(claims.multiply(leads).sum(axis=1)).ravel(),
    }, index=df.index)


def pair_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = tfidf_cosines(df)
    feats["kw_full"] = [keyword_share(k, t) for k, t in zip(df["search_keywords"], df["text_clean"])]
    feats["kw_lead"] = [keyword_share(k, t) for k, t in zip(df["search_keywords"], df["lead_clean"])]
    feats["claim_words_lead"] = [keyword_share(c, t) for c, t in zip(df["claim"], df["lead_clean"])]
    n_words = df["text_clean"].str.split().str.len()
    feats["log_words"] = np.log1p(n_words)
    feats["title_only"] = (n_words < 30).astype(float)
    return feats


# ---------------------------------------------------------------- models
def post_only_model():
    return make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def pair_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


PAIR_COLS = ["cos_full", "cos_lead", "kw_full", "kw_lead", "claim_words_lead", "log_words", "title_only"]


def oof_scores(df: pd.DataFrame, feats: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Out-of-fold scores of every local model for one seed."""
    y = df["relevant"].to_numpy()
    rng = np.random.default_rng(seed)
    scores = {
        "Keyword search (every retrieved post)": rng.random(len(df)) * 1e-9,  # constant score, random ties
        "Keyword overlap": feats["kw_lead"].to_numpy() + feats["kw_full"].to_numpy() / 10,
        "TF-IDF similarity (claim, post)": feats["cos_lead"].to_numpy() + feats["cos_full"].to_numpy(),
        "TF-IDF + logistic regression, post only": np.zeros(len(df)),
        "Pair features + logistic regression": np.zeros(len(df)),
    }
    for train, test in claim_folds(df, y, seed):
        m = post_only_model().fit(df["text_clean"].iloc[train], y[train])
        scores["TF-IDF + logistic regression, post only"][test] = m.predict_proba(df["text_clean"].iloc[test])[:, 1]
        m = pair_model().fit(feats[PAIR_COLS].iloc[train], y[train])
        scores["Pair features + logistic regression"][test] = m.predict_proba(feats[PAIR_COLS].iloc[test])[:, 1]
    return scores


# ---------------------------------------------------------------- metrics
def reading_load(y: np.ndarray, score: np.ndarray, target: float = TARGET_RECALL) -> float:
    order = np.argsort(-score, kind="stable")
    found = np.cumsum(y[order])
    k = int(np.searchsorted(found, np.ceil(target * y.sum()))) + 1
    return k / len(y)


def evaluate(df: pd.DataFrame, scores_by_seed: dict[int, dict[str, np.ndarray]]) -> dict:
    y = df["relevant"].to_numpy()
    per_model: dict[str, dict[str, list]] = {}
    for seed, scores in scores_by_seed.items():
        for train, test in claim_folds(df, y, seed):
            if y[test].sum() == 0:
                continue
            for name, s in scores.items():
                r = per_model.setdefault(name, {"ap": [], "load": []})
                r["ap"].append(average_precision_score(y[test], s[test]))
                r["load"].append(reading_load(y[test], s[test]))
    return {name: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in r.items()}
            for name, r in per_model.items()}


# ---------------------------------------------------------------- transformer scores from Colab
def transformer_scores(df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    """reports/transformers/relevance_scores.csv: post_id, model, seed, score (seed -1: same for all seeds)."""
    if not TRANSFORMER_SCORES.exists():
        return {}
    t = pd.read_csv(TRANSFORMER_SCORES)
    out: dict[int, dict[str, np.ndarray]] = {s: {} for s in SEEDS}
    for (model, seed), g in t.groupby(["model", "seed"]):
        s = df["post_id"].map(g.set_index("post_id")["score"])
        assert s.notna().all(), f"{model}: missing scores"
        for target_seed in (SEEDS if seed == -1 else [seed]):
            if target_seed in out:
                out[target_seed][model] = s.to_numpy()
    return out


def main() -> None:
    df = load_posts()
    y = df["relevant"].to_numpy()
    print(f"{len(df)} posts, {y.sum()} relevant ({y.mean():.1%}), {df['claim_id'].nunique()} claims")
    OUT.mkdir(parents=True, exist_ok=True)

    folds = [pd.DataFrame({"post_id": df["post_id"].iloc[test], "seed": seed, "fold": k})
             for seed in SEEDS for k, (_, test) in enumerate(claim_folds(df, y, seed))]
    pd.concat(folds).sort_values(["seed", "post_id"]).to_csv(DATA / "folds.csv", index=False)

    feats = pair_features(df)
    scores = {seed: oof_scores(df, feats, seed) for seed in SEEDS}
    for seed, extra in transformer_scores(df).items():
        scores[seed].update(extra)
    results = {"dataset": {"posts": len(df), "relevant": int(y.sum()), "claims": int(df["claim_id"].nunique()),
                           "folds": "5 folds grouped by claim", "seeds": list(SEEDS)},
               "models": evaluate(df, scores)}

    # What the pair model relies on (fitted on all posts, standardised features)
    m = pair_model().fit(feats[PAIR_COLS], y)
    results["pair model coefficients"] = dict(zip(PAIR_COLS, np.round(m[-1].coef_[0], 2).tolist()))

    # Errors of the pair model, seed 0: highest-scored off-topic posts and lowest-scored relevant posts
    s0 = scores[0]["Pair features + logistic regression"]
    err = df[["post_id", "claim_id", "stance"]].assign(score=np.round(s0, 3))
    fp = err[err["stance"] == "off_topic"].nlargest(15, "score").assign(error="false positive")
    fn = err[err["stance"] != "off_topic"].nsmallest(15, "score").assign(error="false negative")
    pd.concat([fp, fn]).to_csv(OUT / "errors_seed0.csv", index=False)

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_markdown(results)
    names = list(results["models"])
    rows = [(n.replace(" + ", " +\n", 1) if len(n) > 34 else n, 100 * results["models"][n]["ap"]["mean"],
             100 * results["models"][n]["ap"]["std"], i >= 2) for i, n in enumerate(names)]
    for mode in ("light", "dark"):
        hbar(rows, "Finding the posts that discuss their claim: average precision on unseen claims",
             OUT / f"ap_{mode}.png", mode, ref=(100 * y.mean(), f"share of relevant posts {100 * y.mean():.1f}%"))


def write_markdown(results: dict) -> None:
    d = results["dataset"]
    lines = ["# Relevance results", "",
             f"{d['posts']} posts, {d['relevant']} relevant, {d['claims']} claims. {d['folds']}, "
             f"seeds {d['seeds']}; mean ± std over the test folds.", "",
             f"| Model | Average precision | Posts to read to find {TARGET_RECALL:.0%} of relevant posts |",
             "|---|---|---|"]
    for name, r in results["models"].items():
        lines.append(f"| {name} | {100 * r['ap']['mean']:.1f}% ± {100 * r['ap']['std']:.1f} "
                     f"| {100 * r['load']['mean']:.1f}% ± {100 * r['load']['std']:.1f} |")
    lines += ["", "Coefficients of the pair model (standardised features, fitted on all posts)", ""]
    lines += [f"- `{k}`: {v:+.2f}" for k, v in results["pair model coefficients"].items()]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

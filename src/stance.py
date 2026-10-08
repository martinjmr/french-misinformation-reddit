"""What do the relevant posts say about their claim, and can a model tell?

Input:  data/posts_raw.csv (not versioned), data/claims.csv, data/stance_labels.csv,
        and, when present, reports/transformers/stance_nli.csv (written by the Colab notebook)
Output: reports/stance/{results.json, results.md}

1. Cross the stance labels with the fact-checkers' verdicts: how many posts relay a misleading claim?
2. Predict the stance (supports, refutes, discusses) of the relevant posts.
   Trained models: 5 folds grouped by claim, 3 seeds, predictions pooled over the folds of each seed.
   The zero-shot NLI model is not trained, so it is scored on all relevant posts at once.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

from common import REPORTS, SEEDS, claim_folds, fold_accents, load_posts

OUT = REPORTS / "stance"
NLI_FILE = REPORTS / "transformers" / "stance_nli.csv"
STANCES = ["supports", "refutes", "discusses"]

# Words that signal a debunk or mockery, chosen before looking at the results.
DEBUNK = re.compile(r"\b(faux|fausse|fake|intox|infox|rumeur|complot\w*|conspi\w*|debunk\w*|hoax|bullshit|"
                    r"desinformation|mensonge\w*|n'est pas vrai|aucune preuve|pseudo\w*|arnaque\w*)\b")


def cue_rules(text: str) -> str:
    """Debunk vocabulary -> refutes; a question in the opening words -> discusses; otherwise supports."""
    t = fold_accents(text)
    if DEBUNK.search(t):
        return "refutes"
    if "?" in " ".join(t.split()[:40]):
        return "discusses"
    return "supports"


def text_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def scores(y_true, y_pred) -> dict:
    return {"macro_f1": f1_score(y_true, y_pred, labels=STANCES, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": {s: float(np.mean(np.asarray(y_pred)[np.asarray(y_true) == s] == s)) for s in STANCES},
            "confusion": confusion_matrix(y_true, y_pred, labels=STANCES).tolist()}


def summarise(runs: list[dict]) -> dict:
    out = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
           for k in ("macro_f1", "accuracy")}
    out["recall"] = {s: float(np.mean([r["recall"][s] for r in runs])) for s in STANCES}
    out["confusion (first seed)"] = runs[0]["confusion"]
    return out


def main() -> None:
    df = load_posts()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Stance x verdict on all collected posts
    table = pd.crosstab(df["verdict"], df["stance"]).reindex(columns=["off_topic"] + STANCES, fill_value=0)
    results["stance by verdict"] = table.to_dict(orient="index")

    # 2. Stance prediction on relevant posts
    rel = df[df["relevant"] == 1].reset_index(drop=True)
    y = rel["stance"].to_numpy()
    runs = {"Majority label (supports)": [], "Cue words (rules)": [], "TF-IDF + logistic regression": []}
    for seed in SEEDS:
        pred = np.empty(len(rel), dtype=object)
        for train, test in claim_folds(rel, rel["stance"], seed):
            pred[test] = text_model().fit(rel["text_clean"].iloc[train], y[train]).predict(rel["text_clean"].iloc[test])
        runs["TF-IDF + logistic regression"].append(scores(y, pred))
        runs["Majority label (supports)"].append(scores(y, ["supports"] * len(y)))
        runs["Cue words (rules)"].append(scores(y, rel["text_clean"].map(cue_rules).tolist()))
    results["stance models"] = {name: summarise(r) for name, r in runs.items()}

    if NLI_FILE.exists():  # zero-shot NLI: entailment -> supports, contradiction -> refutes, neutral -> discusses
        nli = rel[["post_id"]].merge(pd.read_csv(NLI_FILE), on="post_id", how="left")
        assert nli["p_entailment"].notna().all(), "missing NLI scores"
        probs = nli[["p_entailment", "p_contradiction", "p_neutral"]].to_numpy()
        pred = np.array(STANCES)[probs.argmax(axis=1)]
        results["stance models"]["Zero-shot NLI (mDeBERTa, no training)"] = summarise([scores(y, pred)])

    results["dataset"] = {"relevant posts": len(rel), "labels": rel["stance"].value_counts().to_dict(),
                          "claims": int(rel["claim_id"].nunique())}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_markdown(results, table)


def write_markdown(results: dict, table: pd.DataFrame) -> None:
    lines = ["# Stance results", "", "## Stance of every collected post, by verdict of its claim", "",
             "| Verdict | " + " | ".join(table.columns) + " | Total |", "|---" * (len(table.columns) + 2) + "|"]
    for verdict, row in table.iterrows():
        lines.append(f"| {verdict} | " + " | ".join(str(v) for v in row) + f" | {row.sum()} |")
    d = results["dataset"]
    lines += ["", "## Predicting the stance of relevant posts", "",
              f"{d['relevant posts']} relevant posts from {d['claims']} claims: {d['labels']}. "
              "Trained models: 5 folds grouped by claim, seeds " + str(list(SEEDS)) + ".", "",
              "| Model | Macro-F1 | Accuracy | Recall supports | Recall refutes | Recall discusses |",
              "|---|---|---|---|---|---|"]
    for name, r in results["stance models"].items():
        lines.append(f"| {name} | {100 * r['macro_f1']['mean']:.1f}% ± {100 * r['macro_f1']['std']:.1f} "
                     f"| {100 * r['accuracy']['mean']:.1f}% | "
                     + " | ".join(f"{100 * r['recall'][s]:.0f}%" for s in STANCES) + " |")
    lines += ["", "Confusion matrices (rows: true stance; columns: " + ", ".join(STANCES) + ")", ""]
    for name, r in results["stance models"].items():
        lines += [f"**{name}**", "", "| | " + " | ".join(STANCES) + " |", "|---" * 4 + "|"]
        lines += [f"| {STANCES[i]} | " + " | ".join(str(x) for x in row) + " |"
                  for i, row in enumerate(r["confusion (first seed)"])]
        lines.append("")
    (OUT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

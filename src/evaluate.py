"""Evaluate the classifier when test claims are seen, and when they are not.

Input:  data/posts_clean.csv (written by preprocess.py)
Output: reports/results.json, reports/results.md, reports/accuracy_{light,dark}.png

Two cross-validation protocols, 5 folds repeated with 5 seeds:
- random split: posts are assigned to folds at random, so posts retrieved
  for the same fact-checked claim end up in both training and test folds;
- unseen claims: all posts of a claim stay in the same fold
  (StratifiedGroupKFold), so test claims never appear in training.

Each protocol scores the text model and two baselines that never read the
text: the label of the post's claim (claim lookup) and the most frequent
label of the post's theme in the training folds (theme only).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["true", "misleading", "unverifiable"]
SEEDS = range(5)
N_FOLDS = 5


def text_model():
    return make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def lookup_baseline(train: pd.DataFrame, test: pd.DataFrame, key: str) -> np.ndarray:
    """Predict the most frequent training label of the test post's claim or theme."""
    table = train.groupby(key)["label"].agg(lambda s: s.value_counts().index[0])
    fallback = train["label"].value_counts().index[0]
    return test[key].map(table).fillna(fallback).to_numpy()


def cross_validate(df: pd.DataFrame, protocol: str) -> dict:
    scores = {"text model": [], "claim lookup": [], "theme only": []}
    pooled_true, pooled_pred = [], []
    for seed in SEEDS:
        if protocol == "random split":
            folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(df, df["label"])
        else:
            folds = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(
                df, df["label"], groups=df["claim_id"])
        for train_idx, test_idx in folds:
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            pred = text_model().fit(train["text_clean"], train["label"]).predict(test["text_clean"])
            scores["text model"].append(np.mean(pred == test["label"]))
            scores["claim lookup"].append(np.mean(lookup_baseline(train, test, "claim_id") == test["label"]))
            scores["theme only"].append(np.mean(lookup_baseline(train, test, "theme") == test["label"]))
            if seed == 0:
                pooled_true += list(test["label"])
                pooled_pred += list(pred)
    summary = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))} for name, v in scores.items()}
    summary["confusion (seed 0, rows = true label)"] = confusion_matrix(
        pooled_true, pooled_pred, labels=LABELS).tolist()
    return summary


def top_terms(df: pd.DataFrame, k: int = 8) -> dict:
    model = text_model().fit(df["text_clean"], df["label"])
    vectorizer, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    vocab = np.array(vectorizer.get_feature_names_out())
    return {label: vocab[np.argsort(clf.coef_[i])[::-1][:k]].tolist() for i, label in enumerate(clf.classes_)}


def plot(results: dict, mode: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    ink = {
        "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#898781",
                      grid="#e1e0d9", accent="#2a78d6", baseline="#898781"),
        "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7", muted="#898781",
                     grid="#2c2c2a", accent="#3987e5", baseline="#898781"),
    }[mode]
    rows = [  # (label, protocol, model, emphasised)
        ("Claim lookup, random split\n(baseline: label of the post's claim)", "random split", "claim lookup", False),
        ("Text model, random split", "random split", "text model", True),
        ("Text model, unseen claims", "unseen claims", "text model", True),
        ("Theme only, unseen claims\n(baseline: no text)", "unseen claims", "theme only", False),
    ]
    values = [100 * results[p][m]["mean"] for _, p, m, _ in rows]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    fig.patch.set_facecolor(ink["surface"])
    ax.set_facecolor(ink["surface"])
    ax.set_xlim(0, 108)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.86, bottom=0.14)
    fig.canvas.draw()

    # 4px rounded data-end, square at the baseline (rounding kept circular on screen)
    bbox = ax.get_window_extent()
    x_per_px = 108 / bbox.width
    y_per_px = len(rows) / bbox.height
    r = 7 * x_per_px
    height = 0.42
    for i, (value, (_, _, _, emphasised)) in enumerate(zip(values, rows)):
        color = ink["accent"] if emphasised else ink["baseline"]
        ax.add_patch(FancyBboxPatch((0, i - height / 2), value, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                    mutation_aspect=y_per_px / x_per_px, linewidth=0, facecolor=color))
        ax.add_patch(Rectangle((0, i - height / 2), max(value - r, 0), height, linewidth=0, facecolor=color))
        ax.text(value + 1.5, i, f"{value:.1f}%", va="center", ha="left", fontsize=9.5,
                color=ink["primary"] if emphasised else ink["secondary"])

    chance = 100 / 3
    ax.axvline(chance, color=ink["muted"], linewidth=1, zorder=0)
    ax.text(chance + 1, -0.62, "chance 33.3%", va="bottom", ha="left", fontsize=8.5, color=ink["muted"])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, *_ in rows], fontsize=9, color=ink["secondary"])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.5, color=ink["muted"])
    ax.tick_params(length=0)
    ax.grid(axis="x", color=ink["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(ink["grid"])
    fig.text(0.02, 0.95, "Accuracy on held-out posts (5-fold CV, 5 seeds)", fontsize=10.5,
             color=ink["primary"], ha="left", va="top", fontweight="bold")
    fig.savefig(path, facecolor=ink["surface"])
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "posts_clean.csv")
    results = {protocol: cross_validate(df, protocol) for protocol in ("random split", "unseen claims")}
    results["top terms"] = top_terms(df)
    results["dataset"] = {"posts": len(df), "claims": int(df["claim_id"].nunique()),
                          "claims per label": df.groupby("label")["claim_id"].nunique().to_dict()}

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for mode in ("light", "dark"):
        plot(results, mode, out / f"accuracy_{mode}.png")

    lines = ["# Results", "", f"{len(df)} posts, {df['claim_id'].nunique()} claims, "
             f"{N_FOLDS}-fold cross-validation repeated with {len(SEEDS)} seeds (mean ± std over folds).", "",
             "| Model | Random split | Unseen claims |", "|---|---|---|"]
    for name in ("text model", "claim lookup", "theme only"):
        cells = [f"{100 * results[p][name]['mean']:.1f}% ± {100 * results[p][name]['std']:.1f}"
                 for p in ("random split", "unseen claims")]
        lines.append(f"| {name.capitalize()} | {cells[0]} | {cells[1]} |")
    lines += ["", "Confusion matrices of the text model (seed 0, rows: true label; columns: "
              + ", ".join(LABELS) + ")", ""]
    for protocol in ("random split", "unseen claims"):
        cm = results[protocol]["confusion (seed 0, rows = true label)"]
        lines += [f"**{protocol.capitalize()}**", "", "| | " + " | ".join(LABELS) + " |", "|---" * 4 + "|"]
        lines += [f"| {LABELS[i]} | " + " | ".join(str(x) for x in row) + " |" for i, row in enumerate(cm)]
        lines.append("")
    lines += ["Terms with the largest weight for each label (model trained on all posts)", ""]
    lines += [f"- **{label}**: {', '.join(terms)}" for label, terms in results["top terms"].items()]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

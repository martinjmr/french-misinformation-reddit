# Stance results

## Check of the labels

On 50 posts labelled blind by the author, the LLM labels agree 94% of the time (Cohen's kappa 0.91); on relevant versus off-topic, 96%.

## Stance of every collected post, by verdict of its claim

| Verdict | off_topic | supports | refutes | discusses | Total |
|---|---|---|---|---|---|
| misleading | 723 | 8 | 15 | 25 | 771 |
| true | 1535 | 79 | 0 | 21 | 1635 |
| unverifiable | 864 | 28 | 10 | 33 | 935 |

## Predicting the stance of relevant posts

219 relevant posts from 40 claims: {'supports': 115, 'discusses': 79, 'refutes': 25}. Trained models: 5 folds grouped by claim, seeds [0, 1, 2].

| Model | Macro-F1 | Accuracy | Recall supports | Recall refutes | Recall discusses |
|---|---|---|---|---|---|
| Majority label (supports) | 23.0% ± 0.0 | 52.5% | 100% | 0% | 0% |
| Cue words (rules) | 47.0% ± 0.0 | 60.3% | 87% | 20% | 34% |
| TF-IDF + logistic regression | 39.4% ± 1.0 | 52.4% | 55% | 4% | 64% |
| Zero-shot NLI (mDeBERTa, no training) | 47.8% ± 0.0 | 50.2% | 37% | 48% | 71% |

Confusion matrices (rows: true stance; columns: supports, refutes, discusses)

**Majority label (supports)**

| | supports | refutes | discusses |
|---|---|---|---|
| supports | 115 | 0 | 0 |
| refutes | 25 | 0 | 0 |
| discusses | 79 | 0 | 0 |

**Cue words (rules)**

| | supports | refutes | discusses |
|---|---|---|---|
| supports | 100 | 2 | 13 |
| refutes | 17 | 5 | 3 |
| discusses | 42 | 10 | 27 |

**TF-IDF + logistic regression**

| | supports | refutes | discusses |
|---|---|---|---|
| supports | 61 | 7 | 47 |
| refutes | 7 | 1 | 17 |
| discusses | 24 | 4 | 51 |

**Zero-shot NLI (mDeBERTa, no training)**

| | supports | refutes | discusses |
|---|---|---|---|
| supports | 42 | 9 | 64 |
| refutes | 1 | 12 | 12 |
| discusses | 9 | 14 | 56 |

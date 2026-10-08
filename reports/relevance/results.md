# Relevance results

3341 posts, 219 relevant, 46 claims. 5 folds grouped by claim, seeds [0, 1, 2]; mean ± std over the test folds.

| Model | Average precision | Posts to read to find 80% of relevant posts |
|---|---|---|
| Keyword search (every retrieved post) | 7.3% ± 1.5 | 80.3% ± 6.5 |
| Keyword overlap | 31.0% ± 7.9 | 40.1% ± 5.8 |
| TF-IDF similarity (claim, post) | 36.8% ± 8.2 | 27.2% ± 10.3 |
| TF-IDF + logistic regression, post only | 15.8% ± 4.5 | 39.6% ± 8.0 |
| Pair features + logistic regression | 35.6% ± 7.4 | 26.0% ± 8.1 |
| CamemBERT fine-tuned on (claim, post) pairs | 21.4% ± 8.1 | 36.4% ± 5.1 |
| Multilingual embeddings (e5), cosine | 43.2% ± 14.2 | 20.4% ± 6.6 |
| Zero-shot NLI (mDeBERTa) | 14.1% ± 8.5 | 73.8% ± 20.1 |

Coefficients of the pair model (standardised features, fitted on all posts)

- `cos_full`: +0.12
- `cos_lead`: +1.40
- `kw_full`: +0.45
- `kw_lead`: +0.48
- `claim_words_lead`: -0.94
- `log_words`: -0.77
- `title_only`: -0.55

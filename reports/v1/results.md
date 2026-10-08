# Version 1 results: predicting the verdict

2307 posts, 46 claims, 5-fold cross-validation repeated with 5 seeds (mean ± std over folds).

| Model | Random split | Unseen claims |
|---|---|---|
| Text model | 65.6% ± 1.9 | 37.6% ± 5.9 |
| Claim lookup | 99.9% ± 0.1 | 31.9% ± 0.9 |
| Theme only | 60.2% ± 1.6 | 36.1% ± 16.9 |

Confusion matrices of the text model (seed 0, rows: true label; columns: true, misleading, unverifiable)

**Random split**

| | true | misleading | unverifiable |
|---|---|---|---|
| true | 535 | 116 | 118 |
| misleading | 119 | 476 | 174 |
| unverifiable | 94 | 157 | 518 |

**Unseen claims**

| | true | misleading | unverifiable |
|---|---|---|---|
| true | 325 | 280 | 164 |
| misleading | 221 | 168 | 380 |
| unverifiable | 137 | 264 | 368 |

Terms with the largest weight for each label (model trained on all posts)

- **misleading**: europe, réforme, plus, retraites, aide, réforme des, aides, retraite
- **true**: jo, record, euros, paris 2024, électricité, pen, le pen, marine
- **unverifiable**: ia, eau, télétravail, réseaux, réseaux sociaux, travail, covid 19, les réseaux

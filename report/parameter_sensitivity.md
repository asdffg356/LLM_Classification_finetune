# Parameter Sensitivity Analysis

This section analyzes the TF-IDF + Logistic Regression quick tuning results. The tuning grid covered max_features values of 30000, 50000, and 100000; ngram_range values of (1, 1) and (1, 2); and C values of 0.05, 0.1, 0.3, 0.5, and 1.0.

## Best Configuration

The best validation result was obtained with:

- max_features = 100000
- ngram_range = (1, 2)
- C = 0.1
- valid_log_loss = 1.080877
- valid_accuracy = 0.385264
- valid_macro_f1 = 0.383576

## Conclusions

- C=0.1 achieved the lowest log loss in the tuning grid. This suggests that a moderate amount of regularization is important for this sparse high-dimensional TF-IDF model.
- When C becomes too large, log loss becomes worse, which indicates that insufficient regularization can make the model over-confident.
- ngram_range=(1, 2) is generally better than unigram-only features, showing that bigram phrase features are useful for capturing response quality patterns.
- max_features=100000 gives the best or a very strong result, suggesting that a larger vocabulary helps capture more text patterns in prompts and responses.
- Therefore, the final tuned baseline uses max_features=100000, ngram_range=(1, 2), and C=0.1.

## Best Per-Parameter Results

- Best C setting: C=0.1, valid_log_loss=1.080877
- Best max_features setting: max_features=100000, valid_log_loss=1.080877
- Best ngram_range setting: ngram_range=(1, 2), valid_log_loss=1.080877

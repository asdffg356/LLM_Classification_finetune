# Error Analysis

This section analyzes validation errors for the final TF-IDF + Logistic Regression model with A/B swap augmentation and swap-test averaging.

## Overall Error Situation

- Total validation samples: 11496
- Correct samples: 4546
- Error samples: 6950
- Accuracy: 0.395442
- Mean prediction confidence: 0.378817
- Mean confidence on correct samples: 0.385297
- Mean confidence on wrong samples: 0.374578

The model is better than random guessing, but the error count is still high. This is expected because the task requires comparing two long answers with nuanced quality differences.

## Class-Level Findings

- A_win accuracy: 0.408921
- B_win accuracy: 0.369117
- tie accuracy: 0.409347

The tie class is difficult because it often requires recognizing that two answers are similarly good or similarly flawed. TF-IDF features mainly capture surface word patterns, so they may not reliably identify subtle equivalence in answer quality.

For true tie samples, the predicted label distribution was:

```text
pred_label_name
A_win    1096
B_win    1002
tie      1454
```

For predicted tie samples, the true label distribution was:

```text
label_name
A_win     971
B_win     954
tie      1454
```

## Response Length Difference

The analysis also examines response_len_diff and abs_response_len_diff. The bin with the lowest accuracy was 200-500 with accuracy 0.369258. The bin with the highest accuracy was 0-200 with accuracy 0.411248.

Large differences in response length can influence the model because TF-IDF features may correlate verbosity with quality. However, longer responses are not always better, and shorter responses are not always worse. This means response_len_diff is useful but also potentially misleading.

## High-Confidence Errors

High-confidence wrong predictions show that the model can be confident for the wrong reasons. These errors indicate that the model has learned useful lexical patterns, but it still does not truly understand answer quality, factual correctness, or deep reasoning.

## Limitations of the TF-IDF Model

- It depends heavily on surface lexical features.
- It is weak at judging factual correctness.
- It is weak at judging deep logical consistency.
- It remains limited for long-text comparison and subjective preference tasks.

Overall, the final TF-IDF model is a strong lightweight baseline, but the error analysis shows why this competition remains challenging and why deeper semantic models may be useful when enough compute and data are available.

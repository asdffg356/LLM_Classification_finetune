# Experiment Results

## 1. TF-IDF + Logistic Regression Initial Baseline

### Method

This experiment uses TF-IDF features and multinomial Logistic Regression for three-class classification. The input text is constructed by concatenating `prompt_clean`, `response_a_clean`, and `response_b_clean`. The model does not use `model_a` or `model_b`, because model names are not available in the test set.

### Parameters

* `max_features = 100000`
* `ngram_range = (1, 2)`
* `C = 2.0`
* `random_state = 42`

### Validation Results

| Metric   |    Value |
| -------- | -------: |
| Log Loss | 1.139320 |
| Accuracy | 0.380219 |
| Macro F1 | 0.380760 |

### Analysis

The initial TF-IDF baseline achieves accuracy and macro F1 higher than random guessing, but its log loss is worse than random three-class prediction, whose theoretical log loss is about `ln(3)=1.0986`. This suggests that the model can capture some lexical patterns, but its probability calibration is poor.

---

## 2. TF-IDF + Logistic Regression Tuned Baseline

### Method

Based on the initial baseline, this experiment tunes the regularization strength and TF-IDF feature settings. The final model still uses only prompt and response text, without using model names.

### Best Parameters

* `max_features = 100000`
* `ngram_range = (1, 2)`
* `C = 0.1`
* `random_state = 42`

### Validation Results

| Metric   |    Value |
| -------- | -------: |
| Log Loss | 1.080877 |
| Accuracy | 0.385264 |
| Macro F1 | 0.383576 |

### Kaggle Public Leaderboard

| Item         |   Value |
| ------------ | ------: |
| Public Score | 1.07737 |
| Public Rank  |     165 |

### Analysis

After parameter tuning, the TF-IDF + Logistic Regression model achieves a validation log loss of 1.080877, which is lower than random three-class prediction. The Kaggle public score is 1.07737, close to the local validation result. This indicates that the local validation split is reasonably consistent with the public test set and that the tuned TF-IDF baseline is a reliable baseline model.

---

## 3. DistilBERT Debug Finetuning

### Method

This experiment uses `distilbert-base-uncased` for sequence classification. It is a small-scale debug experiment, using at most 1500 samples per class. The purpose is to verify that the Transformer finetuning pipeline can run correctly under the local Windows + Conda + GPU environment.

### Settings

* Model: `distilbert-base-uncased`
* Max length: 256
* Samples: at most 1500 per class
* Train rows: 3600
* Validation rows: 900
* Epochs: 2
* Mixed precision: fp16

### Validation Results

| Metric   |    Value |
| -------- | -------: |
| Log Loss | 1.078858 |
| Accuracy | 0.388889 |
| Macro F1 | 0.376489 |

### Analysis

The DistilBERT debug model obtains a slightly lower log loss than the tuned TF-IDF baseline, showing that pretrained language models have potential for this task. However, because the experiment uses only a small subset of the data, the result is not stable enough to serve as the final main model.

---

## 4. DistilBERT Medium Finetuning

### Method

This experiment increases the training set size and uses `distilbert-base-uncased` as a lightweight pretrained Transformer baseline. Each class contains at most 6000 samples. The goal is to compare pretrained language model finetuning with the traditional TF-IDF baseline.

### Settings

* Model: `distilbert-base-uncased`
* Max length: 384
* Samples: at most 6000 per class
* Train rows: 14400
* Validation rows: 3600
* Epochs: 2
* Mixed precision: fp16

### Validation Results

| Metric   |    Value |
| -------- | -------: |
| Log Loss | 1.086948 |
| Accuracy | 0.371944 |
| Macro F1 | 0.371848 |

### Prediction Distribution on Validation Set

| Predicted Class | Count |
| --------------- | ----: |
| B_win           |  1493 |
| A_win           |  1059 |
| tie             |  1048 |

The validation set itself is balanced:

| True Class | Count |
| ---------- | ----: |
| A_win      |  1200 |
| B_win      |  1200 |
| tie        |  1200 |

### Analysis

The DistilBERT medium model does not outperform the tuned TF-IDF baseline. Its validation log loss is 1.086948, higher than the TF-IDF tuned result of 1.080877. This may be caused by several factors. First, the task involves long prompts and long responses, while DistilBERT can only process a limited input length, so truncation may remove important information. Second, DistilBERT is a relatively small pretrained model, and it may not be strong enough to judge subtle differences in answer quality, factual correctness, and reasoning. Third, the task includes a `tie` class, which is more subjective and difficult to learn.

Therefore, DistilBERT medium is kept as a pretrained model comparison experiment, but it is not selected as the final Kaggle submission model.

## 5. TF-IDF + Logistic Regression with A/B Swap Augmentation

### Method

This experiment introduces A/B swap augmentation to reduce possible position bias. For each training sample, `response_a` and `response_b` are exchanged. If the original label is `A_win`, the swapped label becomes `B_win`; if the original label is `B_win`, the swapped label becomes `A_win`; if the original label is `tie`, it remains unchanged.

During inference, swap-test averaging is also used. The model predicts the original test sample and the swapped version of the same sample. The probabilities of the swapped prediction are mapped back to the original A/B order and averaged with the original prediction.

### Validation Results

| Metric | Value |
|---|---:|
| Log Loss | 1.078106 |
| Accuracy | 0.395442 |
| Macro F1 | 0.396300 |

### Kaggle Public Leaderboard

| Version | Method | Public Score | Public Rank |
|---|---|---:|---:|
| Version 1 | TF-IDF + Logistic Regression Tuned | 1.07737 | 165 |
| Version 2 | TF-IDF + Logistic Regression + A/B Swap Augmentation | 1.07454 | 157 |

### Analysis

Compared with the tuned TF-IDF baseline, A/B swap augmentation improves validation log loss from 1.080877 to 1.078106, and also improves accuracy and macro F1. On the Kaggle public leaderboard, the public score improves from 1.07737 to 1.07454. This shows that A/B swap augmentation can reduce response position bias and improve the robustness of the preference classifier.

Therefore, this method is selected as the current final submission model.

---

## Current Conclusion

The current best and final submitted model is:

**TF-IDF + Logistic Regression + A/B Swap Augmentation + Swap-test Averaging**

Validation results:

* Local validation log loss: 1.078106
* Local validation accuracy: 0.395442
* Local validation macro F1: 0.396300

Kaggle public leaderboard results:

* Kaggle public score: 1.07454
* Public rank: 157 / 253

Compared with the tuned TF-IDF baseline, the A/B swap augmentation method improves both local validation performance and Kaggle public leaderboard performance. The public score improves from 1.07737 to 1.07454, and the public rank improves from 165 to 157.

DistilBERT finetuning is kept as a pretrained model comparison experiment. Although the debug version shows that Transformer finetuning can run successfully, the medium version does not outperform the TF-IDF baseline. This suggests that for this task, simple pretrained model finetuning under limited input length and limited training samples may not be sufficient. In contrast, the TF-IDF model with A/B swap augmentation is more stable, easier to reproduce, and currently achieves the best result in this project.

Therefore, the final submitted method of this project is:

**TF-IDF + Logistic Regression with A/B swap augmentation and swap-test averaging.**

---

## 6. Ablation Study of A/B Swap Strategy

### Method

To analyze the contribution of each component in the final method, an ablation study is conducted. The final method consists of three components:

- A: TF-IDF + Logistic Regression tuned baseline
- B: train-time A/B swap augmentation
- C: eval-time swap averaging

Since B and C are both enhancement strategies built on the base classifier, they cannot be used independently without A. Therefore, this experiment compares A, A+B, A+C, and A+B+C.

All experiments use the same train/validation split. The validation set is not augmented, so the comparison is fair.

### Results

| Experiment | Train-time A/B Swap | Eval-time Swap Averaging | Valid Log Loss | Accuracy | Macro F1 |
|---|---|---|---:|---:|---:|
| A | No | No | 1.080877 | 0.385264 | 0.383576 |
| A+B | Yes | No | 1.078106 | 0.395442 | 0.396300 |
| A+C | No | Yes | 1.079799 | 0.388570 | 0.382617 |
| A+B+C | Yes | Yes | 1.078095 | 0.396834 | 0.390631 |

### Analysis

The ablation results show that train-time A/B swap augmentation is the most effective component. Compared with the tuned TF-IDF baseline, A+B reduces validation log loss from 1.080877 to 1.078106 and improves both accuracy and macro F1.

Eval-time swap averaging alone also slightly reduces log loss, but its improvement is smaller than train-time augmentation. The complete method A+B+C achieves the lowest validation log loss, which is the official metric of the Kaggle competition. Although A+B obtains a slightly higher macro F1, A+B+C is selected as the final method because log loss is the primary evaluation metric.

This ablation study confirms that the A/B swap strategy helps reduce response position bias and improves the robustness of the preference classifier.

| 方法                     | 本地 Log Loss | Accuracy | Macro F1 | Kaggle Public |
| ---------------------- | ----------: | -------: | -------: | ------------: |
| TF-IDF + LR 初始版        |    1.139320 | 0.380219 | 0.380760 |             - |
| TF-IDF + LR 调参版        |    1.080877 | 0.385264 | 0.383576 |       1.07737 |
| DistilBERT debug       |    1.078858 | 0.388889 | 0.376489 |             - |
| DistilBERT medium      |    1.086948 | 0.371944 | 0.371848 |             - |
| TF-IDF + LR + A/B swap |    1.078106 | 0.395442 | 0.396300 |       1.07454 |
| 消融 A+B+C               |    1.078095 | 0.396834 | 0.390631 |       1.07454 |


---

## Stage Summary

At this stage, the project has completed data preprocessing, EDA, traditional machine learning baseline experiments, Transformer finetuning experiments, A/B swap augmentation, Kaggle submissions, and ablation study.

The task is formulated as a three-class human preference prediction problem. Given a prompt and two responses, the model predicts whether human preference favors response A, response B, or a tie.

### Main Experimental Findings

First, the initial TF-IDF + Logistic Regression baseline can learn some lexical patterns from the prompt and two responses, but its probability calibration is not good enough. The initial validation log loss is 1.139320, which is worse than random three-class prediction.

After parameter tuning, the TF-IDF + Logistic Regression baseline improves significantly. The best tuned baseline uses:

- `max_features = 100000`
- `ngram_range = (1, 2)`
- `C = 0.1`

The tuned baseline achieves:

| Metric | Value |
|---|---:|
| Validation Log Loss | 1.080877 |
| Accuracy | 0.385264 |
| Macro F1 | 0.383576 |
| Kaggle Public Score | 1.07737 |
| Public Rank | 165 |

This shows that traditional text features are still useful for this task, especially because TF-IDF can capture lexical patterns, response length, common phrases, and structural differences between responses.

Second, DistilBERT finetuning is also tested as a pretrained language model baseline. The debug version verifies that the Transformer finetuning pipeline can run successfully under the local Windows + Conda + GPU environment. However, the larger DistilBERT medium experiment does not outperform the tuned TF-IDF baseline.

The DistilBERT medium result is:

| Metric | Value |
|---|---:|
| Validation Log Loss | 1.086948 |
| Accuracy | 0.371944 |
| Macro F1 | 0.371848 |

This result suggests that simply finetuning a small pretrained model is not necessarily better for this task. Possible reasons include long input truncation, limited training samples, the relatively small capacity of DistilBERT, and the difficulty of predicting the subjective `tie` class.

Third, A/B swap augmentation proves to be the most effective improvement in the current project. By swapping `response_a` and `response_b` during training and swapping labels accordingly, the model can reduce response position bias. In addition, swap-test averaging is used during inference to make the prediction more stable.

The A/B swap version achieves:

| Metric | Value |
|---|---:|
| Validation Log Loss | 1.078106 |
| Accuracy | 0.395442 |
| Macro F1 | 0.396300 |
| Kaggle Public Score | 1.07454 |
| Public Rank | 157 |

Compared with the tuned TF-IDF baseline, the Kaggle public score improves from 1.07737 to 1.07454, and the public rank improves from 165 to 157.

### Ablation Study Conclusion

The ablation study compares four settings:

| Experiment | Components | Validation Log Loss | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| A | TF-IDF + LR | 1.080877 | 0.385264 | 0.383576 |
| A+B | TF-IDF + LR + train-time A/B swap | 1.078106 | 0.395442 | 0.396300 |
| A+C | TF-IDF + LR + eval-time swap averaging | 1.079799 | 0.388570 | 0.382617 |
| A+B+C | TF-IDF + LR + train-time A/B swap + eval-time swap averaging | 1.078095 | 0.396834 | 0.390631 |

The results show that train-time A/B swap augmentation is the most useful component. Eval-time swap averaging also slightly improves log loss. The complete method A+B+C obtains the lowest validation log loss, so it is selected as the final method because log loss is the official metric of the Kaggle competition.

### Current Best Method

The current best method is:

**TF-IDF + Logistic Regression + A/B Swap Augmentation + Swap-test Averaging**

Its current Kaggle public leaderboard result is:

- Public Score: **1.07454**
- Public Rank: **157 / 253**

Although the ranking is not very high, the project has shown a clear improvement path from the initial baseline to the final enhanced method. The experiments also provide interpretable analysis, including parameter tuning, pretrained model comparison, A/B swap augmentation, Kaggle submission, and ablation study.

---

## 7. Parameter Sensitivity Analysis

### Method

To analyze the influence of key hyperparameters in the TF-IDF + Logistic Regression baseline, this project uses the results of the previous grid search. The tested parameters include:

- `max_features`: 30000, 50000, 100000
- `ngram_range`: (1, 1), (1, 2)
- `C`: 0.05, 0.1, 0.3, 0.5, 1.0

The evaluation metric is validation log loss, which is also the official metric of the Kaggle competition. No additional model training is performed in this section; the analysis is based on the saved tuning results.

### Best Configuration

The best validation result is obtained with:

| Parameter | Value |
|---|---:|
| `max_features` | 100000 |
| `ngram_range` | (1, 2) |
| `C` | 0.1 |

The corresponding validation results are:

| Metric | Value |
|---|---:|
| Validation Log Loss | 1.080877 |
| Accuracy | 0.385264 |
| Macro F1 | 0.383576 |

### Analysis of `C`

The regularization parameter `C` has the most obvious influence on validation log loss. In the tuning results, `C=0.1` achieves the lowest log loss. When `C` becomes larger, the validation log loss increases clearly.

Since Logistic Regression is trained on high-dimensional sparse TF-IDF features, weak regularization may cause the model to become over-confident on the validation set. This hurts log loss, because log loss penalizes wrong predictions with high confidence. Therefore, a relatively stronger regularization setting is more suitable for this task.

### Analysis of `max_features`

The value of `max_features` controls the size of the TF-IDF vocabulary. In the experiment, `max_features=100000` achieves the best validation log loss. Compared with 30000 and 50000 features, a larger vocabulary can keep more useful unigram and bigram patterns from prompts and responses.

However, the improvement from 50000 to 100000 is relatively small. This means that increasing the vocabulary size can bring some improvement, but it also increases memory usage and training time. Considering both performance and computational cost, `max_features=100000` is still acceptable in this project.

### Analysis of `ngram_range`

The setting `ngram_range=(1,2)` performs slightly better than unigram-only features `(1,1)`. This indicates that bigram phrase features are useful for this human preference prediction task. Compared with single words, short phrases can better capture response style, common templates, reasoning expressions, and structural patterns.

The improvement is not very large, but because `(1,2)` achieves the best log loss and the training cost is still manageable, the final baseline uses both unigram and bigram TF-IDF features.

### Conclusion

The parameter sensitivity analysis shows that the TF-IDF + Logistic Regression baseline is mainly affected by the regularization strength `C`. The best setting is:

- `max_features = 100000`
- `ngram_range = (1, 2)`
- `C = 0.1`

This configuration achieves a validation log loss of 1.080877 and is used as the tuned TF-IDF baseline. Based on this tuned baseline, the project further introduces A/B swap augmentation and swap-test averaging to construct the final submission model.

---

## 8. Error Analysis

### Overall Error Situation

The final model is evaluated on the validation set. The confusion matrix is shown below:

| True \ Pred | A_win | B_win | tie |
|---|---:|---:|---:|
| A_win | 1641 | 1401 | 971 |
| B_win | 1526 | 1451 | 954 |
| tie | 1096 | 1002 | 1454 |

The class-level accuracy is:

| Class | Accuracy |
|---|---:|
| A_win | 0.4089 |
| B_win | 0.3691 |
| tie | 0.4093 |

The results show that the model performs better than random guessing, but the overall classification accuracy is still limited. The B_win class is the most difficult among the three classes. In addition, there is clear confusion among A_win, B_win, and tie, showing that human preference prediction is a challenging and subjective task.

### Analysis of Tie Samples

The tie class is especially difficult to interpret. Although the accuracy of tie samples is not the lowest, many true tie samples are predicted as A_win or B_win. Similarly, many samples predicted as tie are actually A_win or B_win.

This indicates that the boundary between tie and non-tie is unclear. In many cases, two responses may be similar in surface form, but human annotators still prefer one of them due to subtle differences in helpfulness, safety, factuality, or expression quality. Such differences are difficult for a TF-IDF model to capture.

### Response Length Difference

The analysis also examines the relationship between response length difference and prediction correctness. The mean absolute response length difference is similar between correct and wrong predictions:

| Group | Sample Count | Mean Abs Length Difference | Median Abs Length Difference |
|---|---:|---:|---:|
| Wrong | 6950 | 642.92 | 407 |
| Correct | 4546 | 635.67 | 406 |

The length-difference bucket analysis is:

| Abs Length Difference | Sample Count | Accuracy |
|---|---:|---:|
| 0-200 | 3645 | 0.4112 |
| 200-500 | 2830 | 0.3693 |
| 500-1000 | 2679 | 0.3949 |
| 1000+ | 2342 | 0.4031 |

The accuracy does not change monotonically with response length difference. This suggests that response length is related to prediction, but it is not the decisive factor. A longer response is not always better, and a shorter response is not always worse.

### High-Confidence Errors

Some high-confidence errors occur when two responses are lexically similar. For example, in safety-related prompts such as harmful instruction requests, both responses may contain refusal phrases such as "I can't assist with that" or "I cannot provide instructions." The TF-IDF model may treat the two responses as very similar and predict tie, while the true label may still prefer one response over the other.

This shows that the model can be confident for the wrong reasons. It captures surface lexical similarity, but it cannot fully understand deeper aspects such as factual correctness, safety quality, reasoning quality, and human preference nuance.

### Error Analysis Conclusion

The error analysis shows the main limitation of the final TF-IDF model. It is stable and lightweight, and it benefits from A/B swap augmentation, but it still relies heavily on surface-level lexical features. It is weak at judging deep semantic quality, subtle answer differences, and subjective tie cases. This explains why the model improves over the baseline but still cannot achieve a very high leaderboard ranking.

---

## 9. Extended Experiment: Word-level and Character-level TF-IDF

### Method

Based on the current best TF-IDF + Logistic Regression with A/B swap augmentation, this experiment further adds character-level TF-IDF features. The word-level TF-IDF captures unigram and bigram word features, while the character-level TF-IDF is expected to capture fine-grained patterns such as punctuation, formatting style, code fragments, abbreviations, and template-like expressions.

The final feature representation is constructed by concatenating word-level TF-IDF features and character-level TF-IDF features. Probability smoothing is also tested to reduce over-confident predictions under the log loss metric.

### Validation Results

| Method | Validation Log Loss | Accuracy | Macro F1 |
|---|---:|---:|---:|
| TF-IDF + LR + A/B Swap | 1.078106 | 0.395442 | 0.396300 |
| Word + Char TF-IDF + A/B Swap + Smoothing | 1.078998 | 0.387352 | 0.387994 |

### Analysis

The word-level and character-level TF-IDF extension does not outperform the current best A/B swap model. Although character-level features may capture additional surface patterns such as punctuation, formatting, and code-like text, they also introduce a large number of sparse features. In this experiment, these additional features do not improve validation log loss.

One possible reason is that the original word-level TF-IDF features already capture most useful lexical patterns for this task. Another possible reason is that the Logistic Regression parameter `C=0.1` was tuned on the word-level TF-IDF baseline and may not be optimal for the larger word+char feature space.

Therefore, this method is kept as an extended experiment, but it is not selected as the final submission model.

---

## 10. Final Ensemble Submission

### Method

Based on the previous A/B swap TF-IDF model and the tuned word+char TF-IDF model, this experiment uses weighted probability ensemble as the final submission method.

The final ensemble is:

- 0.7 × TF-IDF + Logistic Regression with A/B swap augmentation and swap-test averaging
- 0.3 × tuned word+char TF-IDF + Logistic Regression with A/B swap augmentation and swap-test averaging

The tuned word+char model uses `C=0.05`, which performs better than the previous `C=0.1` setting.

### Validation Results

| Method | Validation Log Loss | Accuracy | Macro F1 |
|---|---:|---:|---:|
| A/B swap TF-IDF | 1.078106 | 0.395442 | 0.396300 |
| word+char TF-IDF, C=0.05 | 1.078429 | - | - |
| weighted ensemble | 1.077990 | 0.396399 | 0.397240 |

### Kaggle Public Leaderboard

| Version | Method | Public Score |
|---|---|---:|
| Version 1 | TF-IDF + Logistic Regression tuned | 1.07737 |
| Version 2 | TF-IDF + Logistic Regression + A/B swap augmentation | 1.07454 |
| Version 3 | Weighted ensemble of A/B swap TF-IDF and tuned word+char TF-IDF | 1.07430 |

### Analysis

The final ensemble achieves the best local validation log loss and the best Kaggle public score among all submitted versions. Compared with the original TF-IDF baseline, the public score improves from 1.07737 to 1.07430. Compared with the A/B swap model, the improvement is smaller but still positive.

This result shows that the tuned word+char TF-IDF model provides limited but useful complementary information. Although the word+char model alone is not better than the A/B swap TF-IDF model, combining their prediction probabilities slightly improves the final log loss.

Therefore, the final submitted model of this project is selected as:

**Weighted ensemble of A/B swap TF-IDF and tuned word+char TF-IDF, with weights 0.7 and 0.3.**
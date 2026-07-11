# Project Status: LLM Classification Finetuning

## 1. Project task

This repository is for the Kaggle competition **LLM Classification Finetuning**. Given a prompt and two candidate responses, the task is to predict the human preference among three classes:

- `winner_model_a`
- `winner_model_b`
- `winner_tie`

The official metric is multiclass log loss, so every submission must output three probabilities per test row.

## 2. Data scale

Known local training data scale from the project experiments:

- Train rows: 57,477
- Approximate label distribution:
  - A win: 20,064
  - B win: 19,652
  - Tie: 17,761
- Fixed validation split used in local experiments:
  - `test_size=0.2`
  - `random_state=42`
  - `stratify=label`
  - Validation rows: 11,496

Raw Kaggle data, processed data, and large model artifacts are intentionally not committed because of size constraints.

## 3. Completed experiments

### EDA

Completed data reading, stringified JSON/list text parsing, text cleaning, label conversion, length statistics, label distribution analysis, and EDA figures.

Key notebook:

- `notebooks/01_eda.ipynb`

### Initial TF-IDF + Logistic Regression baseline

Input format: prompt, Response A, and Response B concatenated into one text field. Model names are not used as features.

Metrics:

- Validation log loss: 1.139320
- Validation accuracy: 0.380219
- Validation macro F1: 0.380760

Key notebook:

- `notebooks/02_tfidf_baseline.ipynb`

### Tuned word-level TF-IDF baseline

Best parameters:

- `max_features=100000`
- `ngram_range=(1, 2)`
- `C=0.1`

Metrics:

- Validation log loss: 1.080877
- Validation accuracy: 0.385264
- Validation macro F1: 0.383576

Kaggle Version 1 public score:

- 1.07737

Key notebooks and logs:

- `notebooks/02_tfidf_baseline.ipynb`
- `outputs/logs/tfidf_lr_tuning_results.csv`

### DistilBERT experiments

DistilBERT debug result:

- Validation log loss: approximately 1.0788
- Validation accuracy: approximately 0.39

DistilBERT medium result:

- Validation log loss: 1.086948
- Validation accuracy: 0.371944
- Validation macro F1: 0.371848

Conclusion: under the current sample size, truncation, and model scale, DistilBERT did not beat the TF-IDF main line and is kept as a pretrained language model comparison.

Key notebook:

- `notebooks/03_transformer_finetune.ipynb`

### A/B swap augmentation

Training-time A/B swap duplicates the training split with Response A and Response B exchanged, maps A-win to B-win and B-win to A-win, and leaves tie unchanged. Test-time swap averaging predicts both original and swapped orders and maps swapped probabilities back to the original order.

Metrics:

- Validation log loss: 1.078106
- Validation accuracy: 0.395442
- Validation macro F1: 0.396300

Kaggle Version 2 public score:

- 1.07454

Key notebooks:

- `notebooks/05_tfidf_ab_swap_experiment.ipynb`
- `notebooks/06_kaggle_tfidf_ab_swap_submission.ipynb`

### A/B swap ablation

Results:

| Experiment | Description | Validation log loss | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| A | TF-IDF + LR | 1.080877 | 0.385264 | 0.383576 |
| A+B | train-time swap | 1.078106 | 0.395442 | 0.396300 |
| A+C | test-time swap averaging | 1.079799 | 0.388570 | 0.382617 |
| A+B+C | train-time swap + test-time swap averaging | 1.078095 | 0.396834 | 0.390631 |

Key notebook and log:

- `notebooks/07_ablation_ab_swap.ipynb`
- `outputs/logs/ablation_ab_swap_results.csv`

### Parameter sensitivity analysis

Main conclusions from the 30 TF-IDF tuning runs:

- `C=0.1` is best.
- `ngram_range=(1, 2)` is slightly better than `(1, 1)`.
- `max_features=100000` is best, but 50,000 is close.

Key notebook and report:

- `notebooks/08_parameter_sensitivity.ipynb`
- `report/parameter_sensitivity.md`

### Error analysis

Primary object: A/B swap TF-IDF main model.

Validation summary:

- Total validation rows: 11,496
- Correct rows: 4,546
- Incorrect rows: 6,950
- Accuracy: 0.395442

Class-level accuracy:

- A win: approximately 0.4089
- B win: approximately 0.3691
- Tie: approximately 0.4093

Main conclusions:

- B-win is harder to predict.
- Tie is strongly confused with A/B wins.
- Length difference is not decisive.
- TF-IDF cannot directly understand factual correctness, deep logic, safety, or subtle human preferences.

Key notebook and report:

- `notebooks/09_error_analysis.ipynb`
- `report/error_analysis.md`

### Word + char TF-IDF

Initial word+char result with `C=0.1`:

- Validation log loss: 1.078998

After C tuning:

- Best `C=0.05`
- Validation log loss: 1.078429

Conclusion: the single word+char model did not beat the A/B swap main model.

Key notebooks and logs:

- `notebooks/10_tfidf_word_char_ab_swap_smoothing.ipynb`
- `notebooks/12_word_char_c_tuning_and_ensemble.ipynb`
- `outputs/logs/word_char_c_tuning_results.csv`

### Previous final probability ensemble

Blend:

- 0.7 x A/B swap TF-IDF
- 0.3 x tuned word+char TF-IDF

Metrics:

- Validation log loss: 1.077990
- Validation accuracy: 0.396399
- Validation macro F1: 0.397240

Kaggle Version 3 public score:

- 1.07430

This remains the best confirmed Kaggle public score in the repository before submitting the Pairwise model.

Key notebooks:

- `notebooks/11_probability_ensemble.ipynb`
- `notebooks/12_word_char_c_tuning_and_ensemble.ipynb`
- `notebooks/13_kaggle_ensemble_ab_swap_tuned_word_char_submission.ipynb`

## 4. Latest Pairwise TF-IDF experiment

Key notebook:

- `notebooks/14_pairwise_tfidf_difference.ipynb`

The pairwise model uses a shared response TF-IDF space for Response A and Response B and constructs:

- prompt TF-IDF
- `X_diff = X_a - X_b`
- `X_abs_diff = abs(X_a - X_b)`
- `X_sum = 0.5 * (X_a + X_b)`
- 10 numeric difference features

Best local validation result:

- Best pairwise `C`: 0.1
- Best pairwise validation log loss: 1.0178942459584235
- Best temperature: 1.0
- Best pairwise ensemble weight: 1.0
- Best previous-final ensemble weight: 0.0
- Best ensemble validation log loss: 1.017894247085135
- Best ensemble validation accuracy: 0.4880828114126653
- Best ensemble validation macro F1: 0.4828415438001457
- Local pairwise submission generated: true

Interpretation: the pairwise model improves validation log loss by approximately 0.060096 compared with the previous final model. Because the best ensemble assigns weight 1.0 to the pairwise model and 0.0 to the previous model, the improvement mainly comes from the pairwise representation itself rather than complementarity with the old final model.

## 5. Kaggle public scores

Confirmed Kaggle public scores so far:

| Version | Model | Public score |
|---|---|---:|
| 1 | Tuned TF-IDF + Logistic Regression | 1.07737 |
| 2 | TF-IDF + A/B swap | 1.07454 |
| 3 | 0.7 A/B swap TF-IDF + 0.3 tuned word+char TF-IDF | 1.07430 |

Current best confirmed Kaggle public score remains **1.07430**.

## 6. Pairwise Kaggle status

Pairwise Kaggle submission has not yet been confirmed on Kaggle. The formal independent submission notebook is:

- `notebooks/15_kaggle_pairwise_tfidf_submission.ipynb`

It should be uploaded to Kaggle and run from scratch with Internet disabled.

## 7. Current next step

Run `notebooks/15_kaggle_pairwise_tfidf_submission.ipynb` on Kaggle with the official competition dataset attached. Confirm that it creates:

- `/kaggle/working/submission.csv`

Then submit that CSV to Kaggle and record the public score. Do not update final report conclusions until the Pairwise Kaggle score is known.

## 8. Key files to read in a new session

Recommended recovery order:

1. `PROJECT_STATUS.md`
2. `notebooks/14_pairwise_tfidf_difference.ipynb`
3. `report/pairwise_tfidf_experiment.md`
4. `outputs/logs/pairwise_tfidf_c_tuning_results.csv`
5. `outputs/logs/pairwise_temperature_results.csv`
6. `outputs/logs/pairwise_final_ensemble_results.csv`
7. `notebooks/15_kaggle_pairwise_tfidf_submission.ipynb`
8. `report/experiment_results.md`
9. `report/error_analysis.md`
10. `report/parameter_sensitivity.md`

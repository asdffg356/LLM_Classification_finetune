# Pairwise TF-IDF Difference Experiment

## Motivation

The previous models concatenate Prompt, Response A, and Response B into one input. This experiment instead gives both responses a shared TF-IDF space and explicitly represents how they differ. The goal is to expose directional evidence for A versus B while preserving signals useful for tie prediction.

## Feature Design

- `X_diff = X_a - X_b` represents words and phrases that occur more strongly in one response than the other, including direction.
- `X_abs_diff = abs(X_a - X_b)` represents the magnitude of response differences without direction.
- `X_sum = 0.5 * (X_a + X_b)` represents shared overall response content and may help identify ties.
- Prompt TF-IDF and ten scaled numeric difference features provide task context and compact structural asymmetry signals.

## Pairwise Model Result

- Best Logistic Regression C: 0.1
- Validation log loss before calibration: 1.017894
- Validation accuracy: 0.488083
- Validation macro F1: 0.482842

## Temperature Calibration

- Best temperature: 1.0
- Calibrated pairwise validation log loss: 1.017894

## Ensemble with the Current Final Model

- Rebuilt current final model log loss: 1.077990
- Best pairwise weight: 1.00
- Best current-final weight: 0.00
- Best ensemble validation log loss: 1.017894
- Best ensemble validation accuracy: 0.488083
- Best ensemble validation macro F1: 0.482842
- Local sample submission generated: True

## Conclusion

The ensemble improved log loss by 0.060096 relative to 1.077990. This suggests the pairwise representation adds a small amount of complementary information.

Because all components are linear TF-IDF models, their errors can remain highly correlated. Pairwise subtraction improves structural comparison but still cannot directly judge factual correctness, deep reasoning, or nuanced human preference.

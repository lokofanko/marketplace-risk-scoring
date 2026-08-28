# Model Card

## Baseline

`logreg_v1` is a Logistic Regression classifier trained on 5,000 synthetic listing
events. Its saved sklearn `Pipeline` includes one-hot encoding for category/location,
numeric scaling, and the classifier.

The artifact reports ROC-AUC `0.9992`, PR-AUC `0.9978`, and LogLoss `0.0434` on a
synthetic validation split. These values should not be interpreted as real-world
fraud performance because labels are generated from transparent rules that overlap
with model features.

The model is intended to demonstrate reproducible training, artifact serving, and
policy integration. It has not been evaluated on real marketplace traffic, fairness,
adversarial behavior, or temporal drift.

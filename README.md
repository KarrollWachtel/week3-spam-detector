# Email Spam Detector

A small, dependency-free email classifier built around a balanced dataset of 16
labeled examples (8 spam and 8 legitimate).

## Prepare the dataset

`emails.csv` contains the sample email text and labels. Generate
`email_features.csv` with:

```powershell
python prepare_data.py
```

The generated dataset contains only the label and six numeric features. The
classifier never receives the original email text.

## Evaluate and classify

```powershell
python spam_detector.py
python spam_detector.py "URGENT! Claim your cash prize now at https://offer.example"
```

The script makes a reproducible, stratified 75/25 train/test split, trains a
Gaussian Naive Bayes classifier on the training rows, and reports held-out test
accuracy, a four-value confusion matrix, spam precision and recall, feature
importance estimates, a near-duplicate check, and results for three
adversarial examples. If you provide email text, it also trains on the full
sample to make the requested prediction. Use `--spam-threshold` to change the
minimum spam probability required to label a message as spam.

## Run automated tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover deterministic stratified splitting, held-out metrics, feature
importance ranking, threshold validation, and near-duplicate detection. See
[`analysis_report.md`](analysis_report.md) for the recorded results and
interpretation, including the confusion matrix, precision/recall, challenge
emails, and data leakage caveats.

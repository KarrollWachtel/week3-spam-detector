
# Feature engineering

The model receives six numeric features, not the email text: `word_count`,
`exclamations`, `money_words`, `all_caps`, `link_count`, and `urgency_words`.
The 16 examples in `emails.csv` are balanced: 8 spam and 8 legitimate.

## Added features

- **`link_count`:** Counts URLs in the email; spam often directs recipients to
  untrusted pages, while ordinary correspondence is less likely to include
  promotional links.
- **`urgency_words`:** Counts terms such as “urgent,” “now,” and “expires”;
  spam often pressures recipients to act quickly, unlike routine legitimate
  messages.

The dataset builder extracts these values in `prepare_data.py`. The
`GaussianNaiveBayes` classifier trains only on those numeric values in
`spam_detector.py`; it does not inspect email wording directly.

## Training and test split

## 3.1 Confusion matrix

The fixed test split produced these four counts:

| Outcome | Count |
|---|---:|
| Spam correctly caught | 2 |
| Legitimate correctly passed | 2 |
| Legitimate wrongly marked spam | 0 |
| Spam wrongly let through | 0 |

This is 4/4 correct- now,personally- I don't think that my code is 100000% fail proof! If more tests were ran (lets say something crazy like 2,000)I think that there would for sure me some that slip through the cracks!

## 3.2 Precision and recall

- **Spam precision: 100% (2/2).** Of the emails the model called spam, both
  really were spam; none of the legitimate test emails were wrongly flagged.
- **Spam recall: 100% (2/2).** Of the spam emails in the test set, the model
  caught both and let none through.

Similar to above, this is a smaller-scale example. If this example was formatted and made with a much larger audience in mind (like an actual business) there wouldn't be perfect marks. A good example is the company I work for. Our program blocks almost 96% of all spam emails...however, there's always a few that manage to slip through by having the words formatted just right. 

## 3.3 Which error matters more?

To me-I would rather allow an occasional spam message into the inbox
than send a legitimate job offer to spam unnoticed. Letting an actual job offer being sent to spam can cost and cause a huge problem for a company. Accuracy does not distinguish those costs. Raising the spam
decision threshold makes the filter more cautious about calling a message
spam; this can reduce legitimate messages wrongly flagged, at the cost of
letting more spam through. The threshold is configurable, for example:
`python spam_detector.py --spam-threshold 0.8`. On this four-email test split,
the threshold change did not change the measured confusion counts.

## 3.4 What the model learned

The script reports standardized class-mean separation from training data as an
importance estimate (larger means the feature values differ more between
spam and legitimate training examples):

| Rank | Feature | Separation |
|---:|---|---:|
| 1 | `money_words` | 1.939 |
| 2 | `urgency_words` | 1.823 |
| 3 | `exclamations` | 1.667 |
| 4 | `link_count` | 0.833 |
| 5 | `word_count` | 0.527 |
| 6 | `all_caps` | 0.000 |

These are not learned coefficients; Gaussian Naive Bayes has no single linear
weight per feature. Urgency ranks highly, matching the intuition behind that
feature. Link count is less influential than expected, while the original
`money_words` feature is strongest. That difference is useful evidence about
what this particular sample teaches the model.

## 3.5 Adversarial examples

Each challenge email was classified by the model trained on the training set.
All three fooled it:

| Intended label | Challenge email | Prediction | Result |
|---|---|---|---|
| Spam | “Hello, I am following up about your account. A cash refund is available for review at https://account.example when convenient.” | Legitimate | Fooled |
| Legitimate | “URGENT! Please act now: today's team meeting starts at 3 PM. The schedule is attached!” | Spam | Fooled |
| Spam | “Hello, please review the account details at https://account.example when you have a moment.” | Legitimate | Fooled |

The examples show that modest wording changes can hide spam signals, while
legitimate urgent messages can trigger the filter. The classifier only sees
the six engineered numbers, not the context that would distinguish these
messages.

## 3.6 Near-duplicate leakage check

The script compares normalized email text between training and test rows using
`SequenceMatcher`; the reported audit threshold is 80% textual similarity. It
found **0 cross-split pairs** at or above that threshold. A lower 55% check also
found no cross-split pairs for this split. This gives no evidence that a
near-copy crossed the split, but the dataset has no source-family IDs and
string similarity cannot establish whether paraphrased emails came from the
same original. Treat the audit as a useful check, not proof of independence.

Even with no detected near-duplicate leakage, confidence in 100% accuracy
remains low: the dataset has only 16 synthetic emails, just four held out, and
all three hand-crafted challenge examples were misclassified.

## Automated tests

Run the dependency-free test suite with:

```powershell
python -m unittest discover -s tests -v
```

**Recorded result:** 5 tests passed. They verify deterministic stratified
splitting and disjoint rows; confusion-matrix, accuracy, precision, and recall
calculations; feature-importance coverage and ranking; threshold behavior and
validation; and detection of a deliberately similar train/test pair.

## Limitations

Sixteen synthetic examples are enough to demonstrate feature engineering, not
to measure real-world accuracy. Legitimate messages can contain links or urgent
language, and spam can avoid both. The chosen features limit what this model can
learn, and the sample should not be used as a production spam filter.

## Final thoughts

Personally, I think the Spam Detection works really well! And if it were to be applied to a larger-scale
project, I think that it would work well within that enviroment. The models learn based on what default
'urgent' words provided to look for within an email- which since they're given, there's a lower chance for them to pass over them when filtering spam emails. 



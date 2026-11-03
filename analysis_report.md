
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


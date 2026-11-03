import unittest

from prepare_data import FEATURE_NAMES
from spam_detector import (
    GaussianNaiveBayes,
    evaluate,
    feature_importances,
    find_near_duplicate_leaks,
    load_training_examples,
    train_test_split,
)


class SpamDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = load_training_examples()

    def test_split_is_reproducible_stratified_and_disjoint(self):
        training, test = train_test_split(self.examples)
        repeated_training, repeated_test = train_test_split(self.examples)

        self.assertEqual(training, repeated_training)
        self.assertEqual(test, repeated_test)
        self.assertEqual(len(training), 12)
        self.assertEqual(len(test), 4)
        self.assertEqual(
            {label: sum(row["label"] == label for row in test) for label in ("spam", "legitimate")},
            {"spam": 2, "legitimate": 2},
        )
        self.assertFalse(
            {row["_row_id"] for row in training}
            & {row["_row_id"] for row in test}
        )

    def test_evaluation_reports_all_confusion_counts_and_metrics(self):
        result = evaluate(self.examples)

        self.assertEqual(
            result["confusion"],
            {
                "spam_caught": 2,
                "legitimate_passed": 2,
                "legitimate_flagged_as_spam": 0,
                "spam_missed": 0,
            },
        )
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)

    def test_feature_importance_ranks_features_and_covers_all_features(self):
        importances = feature_importances(self.examples)

        self.assertEqual(set(importances), set(FEATURE_NAMES))
        self.assertEqual(next(iter(importances)), "money_words")
        self.assertTrue(all(value >= 0 for value in importances.values()))

    def test_spam_threshold_can_be_configured_and_is_validated(self):
        classifier = GaussianNaiveBayes().fit(self.examples)
        spam_features = next(
            example for example in self.examples if example["label"] == "spam"
        )

        self.assertEqual(classifier.predict(spam_features, spam_threshold=0.5), "spam")
        self.assertEqual(classifier.predict(spam_features, spam_threshold=1.0), "legitimate")
        with self.assertRaises(ValueError):
            classifier.predict(spam_features, spam_threshold=1.1)

    def test_near_duplicate_audit_detects_cross_split_pair(self):
        training = [{"_row_id": 0}]
        test = [{"_row_id": 1}]
        emails = [
            {"email": "A message about a meeting scheduled for Thursday morning."},
            {"email": "A message about a meeting scheduled for Thursday morning!"},
        ]

        leaks = find_near_duplicate_leaks(training, test, emails, threshold=0.8)

        self.assertEqual(len(leaks), 1)
        self.assertEqual((leaks[0][0], leaks[0][1]), (0, 1))


if __name__ == "__main__":
    unittest.main()

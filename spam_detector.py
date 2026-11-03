
import argparse
import csv
from difflib import SequenceMatcher
import math
import random

from prepare_data import EMAILS_PATH, FEATURE_NAMES, FEATURES_PATH, extract_features


TEST_FRACTION = 0.25
RANDOM_SEED = 42
NEAR_DUPLICATE_THRESHOLD = 0.8
ADVERSARIAL_EMAILS = (
    (
        "Low-signal spam written like routine correspondence",
        "spam",
        "Hello, I am following up about your account. A cash refund is available "
        "for review at https://account.example when convenient.",
    ),
    (
        "Legitimate meeting notice with urgent wording",
        "legitimate",
        "URGENT! Please act now: today's team meeting starts at 3 PM. "
        "The schedule is attached!",
    ),
    (
        "Spam with a plain, low-pressure link",
        "spam",
        "Hello, please review the account details at https://account.example "
        "when you have a moment.",
    ),
)


class GaussianNaiveBayes:
    def fit(self, examples):
        if not examples:
            raise ValueError("Cannot train the classifier with an empty dataset.")

        self.classes = sorted({example["label"] for example in examples})
        if self.classes != ["legitimate", "spam"]:
            raise ValueError("Training data must include legitimate and spam labels.")

        pooled_variances = {}
        for feature in FEATURE_NAMES:
            values = [float(example[feature]) for example in examples]
            mean = sum(values) / len(values)
            pooled_variances[feature] = sum(
                (value - mean) ** 2 for value in values
            ) / len(values)

        self.statistics = {}
        for label in self.classes:
            class_examples = [
                example for example in examples if example["label"] == label
            ]
            class_statistics = {}
            for feature in FEATURE_NAMES:
                values = [float(example[feature]) for example in class_examples]
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                class_statistics[feature] = (
                    mean,
                    max(variance, pooled_variances[feature], 1.0),
                )

            self.statistics[label] = {
                "prior": len(class_examples) / len(examples),
                "features": class_statistics,
            }
        return self

    def spam_probability(self, features):
        scores = {}
        for label, statistics in self.statistics.items():
            score = math.log(statistics["prior"])
            for feature in FEATURE_NAMES:
                mean, variance = statistics["features"][feature]
                value = float(features[feature])
                score += -0.5 * math.log(2 * math.pi * variance)
                score -= (value - mean) ** 2 / (2 * variance)
            scores[label] = score

        maximum_score = max(scores.values())
        exp_scores = {
            label: math.exp(score - maximum_score) for label, score in scores.items()
        }
        return exp_scores["spam"] / sum(exp_scores.values())

    def predict(self, features, spam_threshold=0.5):
        if not 0 <= spam_threshold <= 1:
            raise ValueError("The spam threshold must be between 0 and 1.")
        if self.spam_probability(features) >= spam_threshold:
            return "spam"
        return "legitimate"


def train_test_split(examples, test_fraction=TEST_FRACTION, random_seed=RANDOM_SEED):
    examples_by_label = {}
    for example in examples:
        examples_by_label.setdefault(example["label"], []).append(example)

    random_generator = random.Random(random_seed)
    training_examples = []
    test_examples = []
    for label_examples in examples_by_label.values():
        if len(label_examples) < 2:
            raise ValueError("Each label needs at least two examples for a split.")
        shuffled_examples = label_examples.copy()
        random_generator.shuffle(shuffled_examples)
        test_count = max(1, round(len(shuffled_examples) * test_fraction))
        if test_count >= len(shuffled_examples):
            raise ValueError("The test split must leave training examples for each label.")
        test_examples.extend(shuffled_examples[:test_count])
        training_examples.extend(shuffled_examples[test_count:])

    return training_examples, test_examples


def evaluate(examples, spam_threshold=0.5):
    training_examples, test_examples = train_test_split(examples)
    classifier = GaussianNaiveBayes().fit(training_examples)
    confusion = {
        "spam_caught": 0,
        "legitimate_passed": 0,
        "legitimate_flagged_as_spam": 0,
        "spam_missed": 0,
    }
    for example in test_examples:
        prediction = classifier.predict(example, spam_threshold=spam_threshold)
        if example["label"] == "spam":
            confusion["spam_caught" if prediction == "spam" else "spam_missed"] += 1
        else:
            confusion[
                "legitimate_passed"
                if prediction == "legitimate"
                else "legitimate_flagged_as_spam"
            ] += 1

    test_count = len(test_examples)
    correct = confusion["spam_caught"] + confusion["legitimate_passed"]
    spam_predictions = confusion["spam_caught"] + confusion["legitimate_flagged_as_spam"]
    actual_spam = confusion["spam_caught"] + confusion["spam_missed"]
    return {
        "confusion": confusion,
        "accuracy": correct / test_count,
        "precision": confusion["spam_caught"] / spam_predictions if spam_predictions else 0,
        "recall": confusion["spam_caught"] / actual_spam if actual_spam else 0,
        "correct": correct,
        "training_count": len(training_examples),
        "test_count": test_count,
        "training_examples": training_examples,
        "test_examples": test_examples,
    }


def feature_importances(examples):
    examples_by_label = {
        label: [example for example in examples if example["label"] == label]
        for label in ("legitimate", "spam")
    }
    if not examples_by_label["legitimate"] or not examples_by_label["spam"]:
        raise ValueError("Feature importance requires examples from both labels.")

    importances = {}
    for feature in FEATURE_NAMES:
        class_values = {
            label: [float(example[feature]) for example in label_examples]
            for label, label_examples in examples_by_label.items()
        }
        class_means = {
            label: sum(values) / len(values) for label, values in class_values.items()
        }
        all_values = class_values["legitimate"] + class_values["spam"]
        overall_mean = sum(all_values) / len(all_values)
        pooled_variance = sum((value - overall_mean) ** 2 for value in all_values) / len(
            all_values
        )
        importances[feature] = abs(
            class_means["spam"] - class_means["legitimate"]
        ) / max(math.sqrt(pooled_variance), 1.0)
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def find_near_duplicate_leaks(
    training_examples,
    test_examples,
    emails,
    threshold=NEAR_DUPLICATE_THRESHOLD,
):
    training_ids = {example["_row_id"] for example in training_examples}
    test_ids = {example["_row_id"] for example in test_examples}
    leaks = []
    for training_id in training_ids:
        for test_id in test_ids:
            training_text = " ".join(emails[training_id]["email"].lower().split())
            test_text = " ".join(emails[test_id]["email"].lower().split())
            similarity = SequenceMatcher(None, training_text, test_text).ratio()
            if similarity >= threshold:
                leaks.append((training_id, test_id, similarity))
    return leaks


def load_email_texts():
    with EMAILS_PATH.open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_training_examples():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH.name} was not found. Run `python prepare_data.py` first."
        )

    with FEATURES_PATH.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expected_fields = ["label", *FEATURE_NAMES]
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"{FEATURES_PATH.name} must have these columns: "
                f"{', '.join(expected_fields)}."
            )
        return [
            {
                "label": row["label"],
                "_row_id": row_id,
                **{feature: int(row[feature]) for feature in FEATURE_NAMES},
            }
            for row_id, row in enumerate(reader)
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the email classifier and optionally classify an email."
    )
    parser.add_argument("email", nargs="?", help="Email text to classify")
    parser.add_argument(
        "--spam-threshold",
        type=float,
        default=0.5,
        help="Minimum spam probability for a spam prediction (default: 0.5)",
    )
    args = parser.parse_args()

    examples = load_training_examples()
    result = evaluate(examples, spam_threshold=args.spam_threshold)
    print(
        f"Train/test split: {result['training_count']} training, "
        f"{result['test_count']} test "
        f"(stratified 75/25; random seed {RANDOM_SEED})"
    )
    print(f"Spam decision threshold: {args.spam_threshold:.0%}")
    for name, count in result["confusion"].items():
        print(f"{name.replace('_', ' ').capitalize()}: {count}")
    print(
        f"Test accuracy: {result['accuracy']:.1%} "
        f"({result['correct']}/{result['test_count']} correct)"
    )
    print(f"Spam precision: {result['precision']:.1%}")
    print(f"Spam recall: {result['recall']:.1%}")

    importances = feature_importances(result["training_examples"])
    print("Feature importance (standardized class-mean separation):")
    for name, importance in importances.items():
        print(f"  {name}: {importance:.3f}")

    emails = load_email_texts()
    leaks = find_near_duplicate_leaks(
        result["training_examples"], result["test_examples"], emails
    )
    if leaks:
        print(
            f"Near-duplicate train/test pairs (similarity >= "
            f"{NEAR_DUPLICATE_THRESHOLD:.0%}): {len(leaks)}"
        )
        for training_id, test_id, similarity in leaks:
            print(
                f"  rows {training_id + 1} and {test_id + 1}: "
                f"{similarity:.1%} similar"
            )
    else:
        print(
            f"Near-duplicate train/test pairs at "
            f"{NEAR_DUPLICATE_THRESHOLD:.0%} similarity: 0"
        )

    evaluation_classifier = GaussianNaiveBayes().fit(result["training_examples"])
    print("Adversarial examples:")
    for name, expected_label, email_text in ADVERSARIAL_EMAILS:
        prediction = evaluation_classifier.predict(
            extract_features(email_text), spam_threshold=args.spam_threshold
        )
        fooled = prediction != expected_label
        outcome = "fooled" if fooled else "not fooled"
        print(f"  {name}: expected {expected_label}, predicted {prediction} ({outcome})")

    if args.email:
        classifier = GaussianNaiveBayes().fit(examples)
        prediction = classifier.predict(
            extract_features(args.email), spam_threshold=args.spam_threshold
        )
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()

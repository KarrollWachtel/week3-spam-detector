import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EMAILS_PATH = ROOT / "emails.csv"
FEATURES_PATH = ROOT / "email_features.csv"

FEATURE_NAMES = (
    "word_count",
    "exclamations",
    "money_words",
    "all_caps",
    "link_count",
    "urgency_words",
)

WORD_PATTERN = re.compile(r"\b[\w']+\b")
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)
MONEY_WORDS = {
    "bonus",
    "cash",
    "dollar",
    "dollars",
    "money",
    "prize",
    "refund",
    "win",
}
URGENCY_WORDS = {
    "act",
    "deadline",
    "expires",
    "hurry",
    "immediately",
    "now",
    "today",
    "urgent",
}


def extract_features(email_text):
    words = WORD_PATTERN.findall(email_text)
    normalized_words = [word.lower() for word in words]
    return {
        "word_count": len(words),
        "exclamations": email_text.count("!"),
        "money_words": sum(word in MONEY_WORDS for word in normalized_words),
        "all_caps": sum(word.isalpha() and word.isupper() for word in words),
        "link_count": len(URL_PATTERN.findall(email_text)),
        "urgency_words": sum(word in URGENCY_WORDS for word in normalized_words),
    }


def prepare_data():
    with EMAILS_PATH.open("r", newline="", encoding="utf-8") as source:
        emails = list(csv.DictReader(source))

    label_counts = {
        label: sum(row["label"] == label for row in emails)
        for label in ("spam", "legitimate")
    }
    if len(emails) != 16 or label_counts != {"spam": 8, "legitimate": 8}:
        raise ValueError(
            "emails.csv must contain exactly 16 emails: 8 spam and 8 legitimate."
        )

    with FEATURES_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("label", *FEATURE_NAMES))
        writer.writeheader()
        for email in emails:
            writer.writerow(
                {"label": email["label"], **extract_features(email["email"])}
            )

    print(f"Prepared {len(emails)} emails in {FEATURES_PATH.name}.")


if __name__ == "__main__":
    prepare_data()

"""Train the clause risk classifier.

Fine-tunes a small transformer on CUAD clauses mapped to risk levels. The point
is not that a transformer is exotic — it is that a narrow, measurable subtask
belongs in a trained model rather than in a prompt, and that the difference is
measured rather than asserted.

Two baselines are reported alongside it, because "the fine-tuned model works"
means nothing without them:

- **Majority class.** 67% of clauses are medium risk, so any model scoring
  below that has learned nothing.
- **TF-IDF + logistic regression.** If a linear model on word counts matches
  the transformer, the transformer is not earning its cost.

    uv run python ../ml/risk_classifier/train.py --baseline-only
    uv run python ../ml/risk_classifier/train.py --epochs 3
    uv run python ../ml/risk_classifier/train.py --push-to-hub username/model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import RiskLevel, load_cuad  # noqa: E402

LABELS = [RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

#: Small and English-legal-adjacent. InLegalBERT is Indian-legal pretrained but
#: 3x larger; this trains in minutes on CPU and is the honest starting point.
DEFAULT_MODEL = "distilbert-base-uncased"


def split_data(
    clauses: list, test_fraction: float = 0.2, seed: int = 42
) -> tuple[list, list]:
    """Stratified split.

    Stratified because the classes are heavily imbalanced (67/20/13): a random
    split can leave a rare class barely represented in the test set, making the
    reported score depend on the seed.
    """
    import random

    by_class: dict[str, list] = {}
    for clause in clauses:
        by_class.setdefault(clause.risk.value, []).append(clause)

    rng = random.Random(seed)
    train: list = []
    test: list = []
    for group in by_class.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * test_fraction)
        test.extend(shuffled[:cut])
        train.extend(shuffled[cut:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def report(name: str, true: list[str], predicted: list[str]) -> dict[str, float]:
    """Accuracy and macro-F1.

    Macro-F1 rather than accuracy alone: with a 67% majority class, accuracy
    rewards a model that ignores the minority classes entirely — which are
    exactly the high-risk clauses a compliance tool exists to catch.
    """
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    accuracy = accuracy_score(true, predicted)
    macro_f1 = f1_score(true, predicted, average="macro", zero_division=0)
    high_f1 = f1_score(
        true, predicted, labels=["high"], average="macro", zero_division=0
    )

    print(f"\n=== {name} ===")
    print(f"accuracy {accuracy:.4f}   macro-F1 {macro_f1:.4f}   high-risk F1 {high_f1:.4f}")
    print(classification_report(true, predicted, zero_division=0, digits=3))

    return {"accuracy": accuracy, "macro_f1": macro_f1, "high_f1": high_f1}


def majority_baseline(train: list, test: list) -> dict[str, float]:
    most_common = Counter(c.risk.value for c in train).most_common(1)[0][0]
    return report(
        f"baseline: always predict '{most_common}'",
        [c.risk.value for c in test],
        [most_common] * len(test),
    )


def tfidf_baseline(train: list, test: list) -> dict[str, float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000),
        # Balanced weights: without them the linear model collapses onto the
        # majority class, which is not an interesting baseline.
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pipeline.fit([c.text for c in train], [c.risk.value for c in train])
    predicted = pipeline.predict([c.text for c in test])
    return report("baseline: TF-IDF + logistic regression", [c.risk.value for c in test], list(predicted))


def train_transformer(
    train: list, test: list, *, model_name: str, epochs: int, output: Path
) -> dict[str, float]:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class ClauseDataset(Dataset):  # type: ignore[type-arg]
        def __init__(self, clauses: list) -> None:
            self.encodings = tokenizer(
                [c.text for c in clauses],
                truncation=True,
                padding="max_length",
                max_length=256,
            )
            self.labels = [LABEL_TO_ID[c.risk.value] for c in clauses]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict:  # type: ignore[type-arg]
            item = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
    )

    arguments = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=arguments, train_dataset=ClauseDataset(train)
    )
    trainer.train()

    predictions = trainer.predict(ClauseDataset(test))
    predicted = [LABELS[i] for i in np.argmax(predictions.predictions, axis=1)]

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    return report(f"fine-tuned {model_name}", [c.risk.value for c in test], predicted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--push-to-hub", help="HF repo id to upload the model to")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "model"
    )
    args = parser.parse_args()

    print("Loading CUAD...")
    clauses = load_cuad()
    train, test = split_data(clauses)
    distribution = Counter(c.risk.value for c in clauses)
    print(
        f"{len(clauses)} clauses — {dict(distribution)}\n"
        f"train {len(train)}, test {len(test)}"
    )

    started = time.perf_counter()
    results = {
        "majority": majority_baseline(train, test),
        "tfidf": tfidf_baseline(train, test),
    }

    if not args.baseline_only:
        results["transformer"] = train_transformer(
            train, test, model_name=args.model, epochs=args.epochs, output=args.output
        )

    print("\n" + "=" * 62)
    print(f"{'model':<34}{'accuracy':>10}{'macro-F1':>10}{'high-F1':>8}")
    print("-" * 62)
    for name, metrics in results.items():
        print(
            f"{name:<34}{metrics['accuracy']:>10.4f}"
            f"{metrics['macro_f1']:>10.4f}{metrics['high_f1']:>8.4f}"
        )

    if "transformer" in results:
        gain = results["transformer"]["macro_f1"] - results["tfidf"]["macro_f1"]
        print(
            f"\nFine-tuned vs. TF-IDF: {gain:+.4f} macro-F1 "
            f"({'worth its cost' if gain > 0.02 else 'not clearly worth its cost'})"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nMetrics written to {args.output / 'metrics.json'}")
    print(f"Total {time.perf_counter() - started:.1f}s")

    if args.push_to_hub and "transformer" in results:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        AutoModelForSequenceClassification.from_pretrained(args.output).push_to_hub(
            args.push_to_hub
        )
        AutoTokenizer.from_pretrained(args.output).push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

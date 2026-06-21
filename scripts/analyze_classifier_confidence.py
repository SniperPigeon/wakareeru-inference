import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PREDICTION_FILES = [
    Path("/Users/yukun/projects/wakareeru/data/dataset/diagnostics/val_predictions_with_fallback.csv"),
    Path("/Users/yukun/projects/wakareeru/data/dataset/diagnostics/test_predictions_with_fallback.csv"),
]
DEFAULT_THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


@dataclass(frozen=True)
class Prediction:
    confidence: float
    correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze classifier confidence thresholds from prediction CSV files.",
    )
    parser.add_argument(
        "prediction_files",
        nargs="*",
        type=Path,
        default=DEFAULT_PREDICTION_FILES,
        help="Prediction CSV files. Defaults to wakareeru diagnostics val/test files.",
    )
    parser.add_argument(
        "--confidence-column",
        default="confidence",
        help="Column containing top prediction confidence.",
    )
    parser.add_argument(
        "--correct-column",
        default="correct",
        help="Column containing boolean correctness.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(item) for item in DEFAULT_THRESHOLDS),
        help="Comma-separated confidence thresholds to evaluate.",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_predictions(
    path: Path,
    *,
    confidence_column: str,
    correct_column: str,
) -> list[Prediction]:
    predictions = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = {
            column
            for column in [confidence_column, correct_column]
            if column not in (reader.fieldnames or [])
        }
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
        for row in reader:
            predictions.append(
                Prediction(
                    confidence=float(row[confidence_column]),
                    correct=parse_bool(row[correct_column]),
                )
            )
    return predictions


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * probability)
    return sorted_values[index]


def format_float(value: float) -> str:
    return f"{value:.4f}"


def print_quantiles(*, name: str, predictions: list[Prediction]) -> None:
    probabilities = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    print(f"\n{name}")
    print(f"n={len(predictions)}")
    for label, expected in [("correct", True), ("wrong", False)]:
        values = [
            prediction.confidence
            for prediction in predictions
            if prediction.correct is expected
        ]
        rendered = ", ".join(
            f"q{int(probability * 100):02d}={format_float(quantile(values, probability))}"
            for probability in probabilities
        )
        print(f"{label}: n={len(values)} {rendered}")


def print_threshold_sweep(*, predictions: list[Prediction], thresholds: list[float]) -> None:
    total = len(predictions)
    correct_total = sum(prediction.correct for prediction in predictions)
    wrong_total = total - correct_total
    accuracy = correct_total / total if total else 0.0
    print(f"accuracy={format_float(accuracy)} wrong={wrong_total}")
    print(
        "threshold  coverage  accepted_accuracy  abstain_rate  caught_wrong  lost_correct"
    )
    for threshold in thresholds:
        accepted = [
            prediction
            for prediction in predictions
            if prediction.confidence >= threshold
        ]
        abstained = [
            prediction
            for prediction in predictions
            if prediction.confidence < threshold
        ]
        accepted_correct = sum(prediction.correct for prediction in accepted)
        caught_wrong = sum(not prediction.correct for prediction in abstained)
        lost_correct = sum(prediction.correct for prediction in abstained)
        coverage = len(accepted) / total if total else 0.0
        accepted_accuracy = accepted_correct / len(accepted) if accepted else 0.0
        abstain_rate = len(abstained) / total if total else 0.0
        caught_wrong_rate = caught_wrong / wrong_total if wrong_total else 0.0
        lost_correct_rate = lost_correct / correct_total if correct_total else 0.0
        print(
            f"{threshold:>9.2f}  "
            f"{coverage:>8.3f}  "
            f"{accepted_accuracy:>17.3f}  "
            f"{abstain_rate:>12.3f}  "
            f"{caught_wrong_rate:>12.3f}  "
            f"{lost_correct_rate:>12.3f}"
        )


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    combined_predictions = []

    for path in args.prediction_files:
        predictions = load_predictions(
            path,
            confidence_column=args.confidence_column,
            correct_column=args.correct_column,
        )
        combined_predictions.extend(predictions)
        print_quantiles(name=str(path), predictions=predictions)
        print_threshold_sweep(predictions=predictions, thresholds=thresholds)

    if len(args.prediction_files) > 1:
        print_quantiles(name="combined", predictions=combined_predictions)
        print_threshold_sweep(predictions=combined_predictions, thresholds=thresholds)


if __name__ == "__main__":
    main()

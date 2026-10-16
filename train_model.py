import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from ids_model import (
    NSL_KDD_FEATURES,
    build_preprocessor,
    load_nsl_kdd,
    prepare_labels,
)


def score_model(model: Any, features: Any, targets: np.ndarray, labels: list[str]) -> dict:
    raw_predictions = np.asarray(model.predict(features))
    if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1:
        predictions = (raw_predictions.reshape(-1) >= 0.5).astype(int)
    elif raw_predictions.ndim == 2 and raw_predictions.shape[1] > 1:
        predictions = raw_predictions.argmax(axis=1)
    else:
        predictions = raw_predictions.astype(int).reshape(-1)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=np.arange(len(labels))
        ).tolist(),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=np.arange(len(labels)),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def train(dataset_path: Path, output_dir: Path, mode: str, epochs: int, limit: int | None) -> dict:
    data = load_nsl_kdd(dataset_path)
    if limit is not None:
        if limit < 20:
            raise ValueError("--limit must be at least 20.")
        data = data.sample(n=min(limit, len(data)), random_state=42)
    x = data[NSL_KDD_FEATURES]
    labels = prepare_labels(data["label"], mode)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    counts = np.bincount(y)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Training requires at least two classes with two records each.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor(NSL_KDD_FEATURES)
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    output_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    x_train_nn = (
        x_train_t.toarray() if hasattr(x_train_t, "toarray") else np.asarray(x_train_t)
    ).astype(np.float32)
    x_test_nn = (
        x_test_t.toarray() if hasattr(x_test_t, "toarray") else np.asarray(x_test_t)
    ).astype(np.float32)
    class_count = len(label_encoder.classes_)
    inputs = tf.keras.Input(shape=(x_train_nn.shape[1],))
    layer = tf.keras.layers.Dense(128, activation="relu")(inputs)
    layer = tf.keras.layers.Dropout(0.3)(layer)
    layer = tf.keras.layers.Dense(64, activation="relu")(layer)
    layer = tf.keras.layers.Dropout(0.3)(layer)
    layer = tf.keras.layers.Dense(32, activation="relu")(layer)
    outputs = tf.keras.layers.Dense(
        1 if class_count == 2 else class_count,
        activation="sigmoid" if class_count == 2 else "softmax",
    )(layer)
    ann = tf.keras.Model(inputs, outputs)
    ann.compile(
        optimizer="adam",
        loss="binary_crossentropy" if class_count == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = ann.fit(
        x_train_nn,
        y_train if class_count > 2 else y_train.astype("float32"),
        validation_split=0.2,
        epochs=epochs,
        batch_size=256,
        callbacks=[callback],
        verbose=2,
    )

    baselines = {
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
    }
    ann_metrics = score_model(ann, x_test_nn, y_test, list(label_encoder.classes_))
    comparison = {"ann": ann_metrics}
    for name, baseline in baselines.items():
        baseline.fit(x_train_t, y_train)
        comparison[name] = score_model(
            baseline, x_test_t, y_test, list(label_encoder.classes_)
        )
        joblib.dump(baseline, output_dir / f"{name}.joblib")

    ann.save(output_dir / "ann_model.keras")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
    (output_dir / "feature_columns.json").write_text(
        json.dumps(NSL_KDD_FEATURES, indent=2), encoding="utf-8"
    )
    metrics = {
        "mode": mode,
        "classes": label_encoder.classes_.tolist(),
        "dataset": str(dataset_path),
        "training_records": int(len(x_train)),
        "test_records": int(len(x_test)),
        "comparison": comparison,
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ANN, Random Forest and Logistic Regression on NSL-KDD."
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets/KDDTrain+.txt"))
    parser.add_argument("--output", type=Path, default=Path("saved_models"))
    parser.add_argument("--mode", choices=("binary", "multiclass"), default="binary")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--limit", type=int, help="Optional reproducible record cap for quick runs.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    result = train(args.dataset, args.output, args.mode, args.epochs, args.limit)
    print(
        f"Training complete: {result['training_records']} train records, "
        f"{result['test_records']} test records. Artifacts saved to {args.output}."
    )


if __name__ == "__main__":
    main()

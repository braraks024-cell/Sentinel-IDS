import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
NSL_KDD_COLUMNS = NSL_KDD_FEATURES + ["label", "difficulty"]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

ATTACK_CATEGORIES = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L",
    "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L",
    "named": "R2L", "worm": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def map_attack_category(label: object) -> str:
    normalized = str(label).strip().rstrip(".").lower()
    return ATTACK_CATEGORIES.get(normalized, "Other")


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [name for name in CATEGORICAL_FEATURES if name in feature_columns]
    numerical = [name for name in feature_columns if name not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numerical:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numerical))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise ValueError("At least one feature column is required.")
    return ColumnTransformer(transformers, remainder="drop")


def validate_feature_frame(
    data: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    missing = [name for name in feature_columns if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("The input data contains no records.")
    frame = data.loc[:, feature_columns].copy()
    for name in frame.columns:
        if name not in CATEGORICAL_FEATURES:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
            if frame[name].isna().all():
                raise ValueError(f"Numeric feature '{name}' has no valid values.")
    return frame


def load_nsl_kdd(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS, skipinitialspace=True)
    if data.empty:
        raise ValueError(f"Dataset is empty: {path}")
    data["label"] = data["label"].astype(str).str.strip().str.rstrip(".")
    return data


def prepare_labels(labels: pd.Series, mode: str) -> np.ndarray:
    normalized = labels.astype(str).str.strip().str.rstrip(".").str.lower()
    if mode == "binary":
        return np.where(normalized == "normal", "Normal", "Attack")
    if mode == "multiclass":
        mapped = labels.map(map_attack_category)
        if (mapped == "Other").any():
            unknown = sorted(labels[mapped == "Other"].astype(str).unique())
            raise ValueError(f"Unmapped attack labels in dataset: {', '.join(unknown)}")
        return mapped.to_numpy()
    raise ValueError("Mode must be 'binary' or 'multiclass'.")


class IDSModel:
    def __init__(
        self,
        feature_columns: list[str],
        preprocessor: Any,
        label_encoder: LabelEncoder,
        model: Any,
        mode: str,
    ) -> None:
        self.feature_columns = feature_columns
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.model = model
        self.mode = mode

    @classmethod
    def from_artifacts(cls, model_dir: str | Path) -> "IDSModel | None":
        directory = Path(model_dir)
        model_path = directory / "random_forest.joblib"
        if not model_path.exists():
            return None
        with (directory / "feature_columns.json").open(encoding="utf-8") as handle:
            feature_columns = json.load(handle)
        preprocessor = joblib.load(directory / "preprocessor.joblib")
        label_encoder = joblib.load(directory / "label_encoder.joblib")
        model = joblib.load(model_path)
        metrics_path = directory / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return cls(feature_columns, preprocessor, label_encoder, model, metrics["mode"])

    def predict(self, data: pd.DataFrame) -> list[dict[str, object]]:
        frame = validate_feature_frame(data, self.feature_columns)
        transformed = self.preprocessor.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray().astype(np.float32)
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(transformed))
        else:
            probabilities = np.asarray(self.model.predict(transformed, verbose=0))
        if probabilities.ndim == 1 or probabilities.shape[-1] == 1:
            positive = probabilities.reshape(-1)
            indices = (positive >= 0.5).astype(int)
            confidence = np.where(indices == 1, positive, 1 - positive)
        else:
            indices = probabilities.argmax(axis=1)
            confidence = probabilities[np.arange(len(indices)), indices]
        labels = self.label_encoder.inverse_transform(indices)
        result = []
        for label, score in zip(labels, confidence):
            label_text = str(label)
            category = map_attack_category(label_text) if self.mode == "multiclass" else (
                "Normal" if label_text == "Normal" else "Attack"
            )
            result.append(
                {
                    "predicted_label": label_text,
                    "attack_category": category,
                    "confidence": float(score),
                }
            )
        return result

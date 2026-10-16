import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ids_model import IDSModel, NSL_KDD_COLUMNS, NSL_KDD_FEATURES
from train_model import train


class TrainingIntegrationTests(unittest.TestCase):
    def test_train_synthetic_nsl_kdd_and_reload_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for index in range(40):
                label = "normal." if index % 2 == 0 else "neptune."
                row = {
                    name: float((index % 7) + (0 if label == "normal." else 5))
                    for name in NSL_KDD_FEATURES
                }
                row.update(
                    {
                        "protocol_type": "tcp" if index % 3 else "udp",
                        "service": "http" if label == "normal." else "private",
                        "flag": "SF" if label == "normal." else "S0",
                        "label": label,
                        "difficulty": 0,
                    }
                )
                rows.append(row)
            dataset = root / "KDDTrain+.txt"
            pd.DataFrame(rows, columns=NSL_KDD_COLUMNS).to_csv(
                dataset, index=False, header=False
            )

            output_dir = root / "artifacts"
            metrics = train(dataset, output_dir, "binary", epochs=1, limit=None)

            self.assertEqual(metrics["mode"], "binary")
            self.assertEqual(metrics["test_records"], 8)
            self.assertEqual(
                set(metrics["comparison"]),
                {"ann", "random_forest", "logistic_regression"},
            )
            for artifact in (
                "ann_model.keras",
                "preprocessor.joblib",
                "label_encoder.joblib",
                "feature_columns.json",
                "metrics.json",
                "random_forest.joblib",
                "logistic_regression.joblib",
            ):
                self.assertTrue((output_dir / artifact).is_file(), artifact)

            predictor = IDSModel.from_artifacts(output_dir)
            self.assertIsNotNone(predictor)
            prediction = predictor.predict(
                pd.DataFrame([rows[0]]).loc[:, NSL_KDD_FEATURES]
            )
            self.assertEqual(len(prediction), 1)
            self.assertIn(prediction[0]["predicted_label"], ("Normal", "Attack"))
            self.assertGreaterEqual(prediction[0]["confidence"], 0)
            self.assertLessEqual(prediction[0]["confidence"], 1)


if __name__ == "__main__":
    unittest.main()

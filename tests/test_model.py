import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ids_model import (
    build_preprocessor,
    map_attack_category,
    prepare_labels,
    validate_feature_frame,
)
from train_model import score_model


class ModelUtilityTests(unittest.TestCase):
    def test_nsl_kdd_categories_and_binary_mapping(self) -> None:
        self.assertEqual(map_attack_category("normal."), "Normal")
        self.assertEqual(map_attack_category("neptune."), "DoS")
        self.assertEqual(map_attack_category("ipsweep"), "Probe")
        self.assertEqual(map_attack_category("guess_passwd"), "R2L")
        self.assertEqual(map_attack_category("buffer_overflow"), "U2R")
        self.assertEqual(
            prepare_labels(pd.Series(["normal.", "neptune."]), "binary").tolist(),
            ["Normal", "Attack"],
        )

    def test_feature_validation_reports_missing_columns_and_empty_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required feature columns: protocol_type"):
            validate_feature_frame(pd.DataFrame({"duration": [1]}), ["duration", "protocol_type"])
        with self.assertRaisesRegex(ValueError, "contains no records"):
            validate_feature_frame(
                pd.DataFrame(columns=["duration", "protocol_type"]),
                ["duration", "protocol_type"],
            )

    def test_preprocessor_handles_missing_values_and_unseen_categories(self) -> None:
        train = pd.DataFrame(
            {
                "duration": [0, 1, None, 4],
                "protocol_type": ["tcp", "udp", "tcp", "icmp"],
            }
        )
        preprocessor = build_preprocessor(["duration", "protocol_type"])
        training_features = preprocessor.fit_transform(train)
        prediction_features = preprocessor.transform(
            pd.DataFrame({"duration": [2], "protocol_type": ["unknown"]})
        )
        self.assertEqual(training_features.shape[0], 4)
        self.assertEqual(prediction_features.shape[0], 1)

    def test_score_model_converts_binary_probabilities_to_classes(self) -> None:
        model = LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        metrics = score_model(model, [[0], [3]], np.array([0, 1]), ["Attack", "Normal"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()

import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from werkzeug.security import check_password_hash

import database
from app import create_app
from ids_model import validate_feature_frame


class StubPredictor:
    feature_columns = ["duration", "protocol_type"]

    def predict(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        frame = validate_feature_frame(frame, self.feature_columns)
        records = []
        for row in frame.to_dict(orient="records"):
            label = "Attack" if float(row["duration"]) >= 5 else "Normal"
            records.append(
                {
                    "predicted_label": label,
                    "attack_category": label,
                    "confidence": 0.9,
                }
            )
        return records


class AppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.app = create_app(
            {
                "TESTING": True,
                "WTF_CSRF_ENABLED": False,
                "SECRET_KEY": "test-only-secret",
                "DATABASE": str(root / "database.sqlite3"),
                "UPLOAD_DIR": root / "uploads",
                "MODEL_DIR": root / "models",
                "PREDICTOR": StubPredictor(),
                "MAX_CONTENT_LENGTH": 1024,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def register(self, username: str = "analyst", email: str = "analyst@example.test") -> None:
        response = self.client.post(
            "/register",
            data={"username": username, "email": email, "password": "a-secure-pass"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

    def login(self, username: str = "analyst") -> None:
        response = self.client.post(
            "/login",
            data={"username": username, "password": "a-secure-pass"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

    def test_health_and_protected_route(self) -> None:
        self.assertEqual(self.client.get("/health").json, {"status": "ok"})
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_registration_hashes_password_and_login_works(self) -> None:
        self.register()
        with self.app.app_context():
            user = database.get_db().execute(
                "SELECT password_hash FROM users WHERE username = ?", ("analyst",)
            ).fetchone()
        self.assertNotEqual(user["password_hash"], "a-secure-pass")
        self.assertTrue(check_password_hash(user["password_hash"], "a-secure-pass"))
        self.login()
        self.assertEqual(self.client.get("/dashboard").status_code, 200)

    def test_duplicate_registration_is_reported(self) -> None:
        self.register()
        response = self.client.post(
            "/register",
            data={"username": "analyst", "email": "different@example.test", "password": "a-secure-pass"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"already registered", response.data)

    def test_invalid_login_does_not_authenticate(self) -> None:
        self.register()
        response = self.client.post(
            "/login",
            data={"username": "analyst", "password": "incorrect-password"},
        )
        self.assertIn(b"Invalid username or password", response.data)
        self.assertEqual(self.client.get("/dashboard").status_code, 302)

    def test_upload_persists_predictions_and_exports_csv(self) -> None:
        self.register()
        self.login()
        response = self.client.post(
            "/upload",
            data={
                "file": (
                    io.BytesIO(b"duration,protocol_type\n0,tcp\n7,udp\n"),
                    "traffic.csv",
                )
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analyzed 2 records", response.data)
        with self.app.app_context():
            analysis = database.get_db().execute(
                "SELECT * FROM analyses"
            ).fetchone()
            predictions = database.get_db().execute(
                "SELECT predicted_label FROM predictions ORDER BY record_index"
            ).fetchall()
        self.assertEqual(analysis["total_records"], 2)
        self.assertEqual(analysis["normal_count"], 1)
        self.assertEqual([row["predicted_label"] for row in predictions], ["Normal", "Attack"])
        report = self.client.get(f"/reports.csv?analysis_id={analysis['id']}")
        self.assertEqual(report.status_code, 200)
        self.assertIn(b"record_id,prediction,category,confidence,timestamp", report.data)
        self.assertIn(b"Attack", report.data)
        self.assertEqual(list((Path(self.temp_dir.name) / "uploads").iterdir()), [])

    def test_wrong_upload_type_is_rejected(self) -> None:
        self.register()
        self.login()
        response = self.client.post(
            "/upload",
            data={"file": (io.BytesIO(b"not a csv"), "traffic.exe")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertIn(b"Only .csv files are accepted", response.data)

    def test_missing_feature_header_is_reported(self) -> None:
        self.register()
        self.login()
        response = self.client.post(
            "/upload",
            data={"file": (io.BytesIO(b"duration\n1\n"), "traffic.csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertIn(b"Missing required feature columns: protocol_type", response.data)

    def test_history_and_reports_are_private_to_each_user(self) -> None:
        self.register()
        self.login()
        self.client.post(
            "/upload",
            data={"file": (io.BytesIO(b"duration,protocol_type\n0,tcp\n"), "mine.csv")},
            content_type="multipart/form-data",
        )
        self.client.post("/logout")
        self.register("another", "another@example.test")
        self.login("another")
        self.assertIn(b"No analyses found", self.client.get("/history").data)
        report = self.client.get("/reports.csv")
        self.assertEqual(report.data.count(b"record_id"), 1)
        with self.app.app_context():
            analysis_id = database.get_db().execute(
                "SELECT id FROM analyses LIMIT 1"
            ).fetchone()["id"]
        private_report = self.client.get(f"/reports.csv?analysis_id={analysis_id}")
        self.assertEqual(private_report.data.count(b"record_id"), 1)

    def test_upload_size_limit_returns_clear_error(self) -> None:
        self.register()
        self.login()
        response = self.client.post(
            "/upload",
            data={"file": (io.BytesIO(b"x" * 2048), "large.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn(b"Upload exceeds the 10 MB limit", response.data)

    def test_missing_model_shows_instructions_and_blocks_prediction(self) -> None:
        self.client.post(
            "/register",
            data={"username": "analyst", "email": "analyst@example.test", "password": "a-secure-pass"},
        )
        self.login()
        app_without_model = create_app(
            {
                "TESTING": True,
                "WTF_CSRF_ENABLED": False,
                "SECRET_KEY": "test-only-secret",
                "DATABASE": str(Path(self.temp_dir.name) / "database.sqlite3"),
                "UPLOAD_DIR": Path(self.temp_dir.name) / "uploads",
                "MODEL_DIR": Path(self.temp_dir.name) / "models",
                "PREDICTOR": None,
            }
        )
        client_without_model = app_without_model.test_client()
        with client_without_model.session_transaction() as session:
            session["user_id"] = 1
        evaluation = client_without_model.get("/evaluation")
        self.assertIn(b"Model not trained yet", evaluation.data)
        response = client_without_model.get("/predict/manual", follow_redirects=True)
        self.assertIn(b"Train it with", response.data)


if __name__ == "__main__":
    unittest.main()

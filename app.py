import csv
import io
import json
import re
import sqlite3
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import database
from config import Config
csrf = CSRFProtect()
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    app.config["MODEL_DIR"] = Path(app.config["MODEL_DIR"])
    app.config["UPLOAD_DIR"] = Path(app.config["UPLOAD_DIR"])
    app.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["MODEL_DIR"].mkdir(parents=True, exist_ok=True)
    app.teardown_appcontext(database.close_db)
    csrf.init_app(app)
    with app.app_context():
        database.init_db()

    predictor = app.config.get("PREDICTOR")
    if "PREDICTOR" not in app.config:
        from ids_model import IDSModel

        predictor = IDSModel.from_artifacts(app.config["MODEL_DIR"])
    app.extensions["ids_predictor"] = predictor

    @app.before_request
    def load_user() -> None:
        g.user = None
        session_user_id = session.get("user_id")
        if session_user_id is not None:
            g.user = database.get_db().execute(
                "SELECT id, username, email FROM users WHERE id = ?",
                (session_user_id,),
            ).fetchone()
            if g.user is None:
                session.clear()

    @app.context_processor
    def inject_globals() -> dict:
        return {"model_ready": app.extensions["ids_predictor"] is not None}

    @app.errorhandler(413)
    def request_too_large(_error: Exception) -> tuple[str, int]:
        flash("Upload exceeds the 10 MB limit.", "error")
        return render_template("upload.html"), 413

    def login_required(view):
        from functools import wraps

        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.user is None:
                flash("Please log in to continue.", "error")
                return redirect(url_for("login"))
            return view(*args, **kwargs)

        return wrapped

    @app.get("/")
    def index():
        return redirect(url_for("dashboard" if g.user else "login"))

    @app.route("/register", methods=("GET", "POST"))
    def register():
        if g.user:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            if not 3 <= len(username) <= 40:
                flash("Username must contain between 3 and 40 characters.", "error")
            elif not EMAIL_PATTERN.fullmatch(email):
                flash("Enter a valid email address.", "error")
            elif len(password) < 8:
                flash("Password must contain at least 8 characters.", "error")
            else:
                connection = database.get_db()
                try:
                    connection.execute(
                        "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                        (username, email, generate_password_hash(password)),
                    )
                    connection.commit()
                except Exception as error:
                    if isinstance(error, sqlite3.IntegrityError) and "UNIQUE constraint failed" in str(error):
                        flash("That username or email is already registered.", "error")
                    else:
                        raise
                else:
                    flash("Account created. Please log in.", "success")
                    return redirect(url_for("login"))
        return render_template("register.html")

    @app.route("/login", methods=("GET", "POST"))
    def login():
        if g.user:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            user = database.get_db().execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            if user is None or not check_password_hash(user["password_hash"], password):
                flash("Invalid username or password.", "error")
            else:
                session.clear()
                session["user_id"] = user["id"]
                destination = request.args.get("next", "")
                if destination.startswith("/") and not destination.startswith("//"):
                    return redirect(destination)
                return redirect(url_for("dashboard"))
        return render_template("login.html")

    @app.post("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "success")
        return redirect(url_for("login"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        connection = database.get_db()
        totals = connection.execute(
            """SELECT COALESCE(SUM(total_records), 0) AS total,
                      COALESCE(SUM(normal_count), 0) AS normal,
                      COALESCE(SUM(attack_count), 0) AS attacks
               FROM analyses WHERE user_id = ?""",
            (g.user["id"],),
        ).fetchone()
        recent = connection.execute(
            """SELECT a.*, (SELECT p.predicted_label FROM predictions p
               WHERE p.analysis_id = a.id ORDER BY p.id DESC LIMIT 1) AS last_label
               FROM analyses a WHERE a.user_id = ?
               ORDER BY a.created_at DESC, a.id DESC LIMIT 8""",
            (g.user["id"],),
        ).fetchall()
        return render_template("dashboard.html", totals=totals, recent=recent)

    @app.route("/upload", methods=("GET", "POST"))
    @login_required
    def upload():
        if request.method == "POST":
            predictor = app.extensions["ids_predictor"]
            if predictor is None:
                flash("Model not trained yet. Train it with `python train_model.py` first.", "error")
                return redirect(url_for("evaluation"))
            file = request.files.get("file")
            if file is None or not file.filename:
                flash("Choose a CSV file to analyze.", "error")
                return redirect(url_for("upload"))
            if Path(file.filename).suffix.lower() != ".csv":
                flash("Only .csv files are accepted.", "error")
                return redirect(url_for("upload"))

            safe_name = secure_filename(file.filename)
            stored_name = f"{uuid.uuid4().hex}_{safe_name or 'traffic.csv'}"
            destination = Path(app.config["UPLOAD_DIR"]) / stored_name
            try:
                file.save(destination)
                frame = pd.read_csv(destination)
                results = predictor.predict(frame)
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
                flash(f"Could not analyze CSV: {error}", "error")
                return redirect(url_for("upload"))
            finally:
                destination.unlink(missing_ok=True)

            extra_columns = [
                name for name in frame.columns
                if name not in predictor.feature_columns and name not in ("label", "difficulty")
            ]
            analysis_id = database.save_analysis(
                g.user["id"], "csv", safe_name or "traffic.csv", results
            )
            flash(f"Analyzed {len(results)} records.", "success")
            return render_template(
                "results.html",
                analysis_id=analysis_id,
                results=results[:100],
                total=len(results),
                extra_columns=extra_columns,
            )
        return render_template("upload.html")

    @app.route("/predict/manual", methods=("GET", "POST"))
    @login_required
    def manual_prediction():
        predictor = app.extensions["ids_predictor"]
        if predictor is None:
            flash("Model not trained yet. Train it with `python train_model.py` first.", "error")
            return redirect(url_for("evaluation"))
        values = {}
        result = None
        if request.method == "POST":
            values = request.form.to_dict()
            try:
                frame = pd.DataFrame([{name: values.get(name, "") for name in predictor.feature_columns}])
                predictions = predictor.predict(frame)
                result = predictions[0]
                database.save_analysis(g.user["id"], "manual", None, predictions)
            except ValueError as error:
                flash(str(error), "error")
        return render_template(
            "manual.html",
            feature_columns=predictor.feature_columns,
            categorical_features={"protocol_type", "service", "flag"},
            values=values,
            result=result,
        )

    @app.get("/evaluation")
    @login_required
    def evaluation():
        metrics_path = Path(app.config["MODEL_DIR"]) / "metrics.json"
        metrics = None
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return render_template("evaluation.html", metrics=metrics)

    @app.get("/history")
    @login_required
    def history():
        page = max(request.args.get("page", 1, type=int), 1)
        per_page = 25
        date_filter = request.args.get("date", "")
        label_filter = request.args.get("label", "")
        where = ["a.user_id = ?"]
        params: list[object] = [g.user["id"]]
        if date_filter:
            try:
                date.fromisoformat(date_filter)
            except ValueError:
                abort(400, "Date filter must use YYYY-MM-DD format.")
            where.append("date(a.created_at) = ?")
            params.append(date_filter)
        if label_filter in ("Normal", "Attack", "DoS", "Probe", "R2L", "U2R", "Other"):
            where.append(
                "EXISTS (SELECT 1 FROM predictions p WHERE p.analysis_id = a.id "
                "AND (p.predicted_label = ? OR p.attack_category = ?))"
            )
            params.extend((label_filter, label_filter))
        conditions = " AND ".join(where)
        connection = database.get_db()
        total = connection.execute(
            f"SELECT COUNT(*) FROM analyses a WHERE {conditions}", params
        ).fetchone()[0]
        rows = connection.execute(
            f"""SELECT a.* FROM analyses a WHERE {conditions}
                ORDER BY a.created_at DESC, a.id DESC LIMIT ? OFFSET ?""",
            (*params, per_page, (page - 1) * per_page),
        ).fetchall()
        return render_template(
            "history.html",
            rows=rows,
            page=page,
            pages=max((total + per_page - 1) // per_page, 1),
            date_filter=date_filter,
            label_filter=label_filter,
        )

    @app.get("/reports.csv")
    @login_required
    def export_report():
        analysis_id = request.args.get("analysis_id", type=int)
        query = """SELECT p.id, p.predicted_label, p.attack_category,
                          p.confidence, p.created_at
                   FROM predictions p JOIN analyses a ON a.id = p.analysis_id
                   WHERE a.user_id = ?"""
        params: list[object] = [g.user["id"]]
        if analysis_id is not None:
            query += " AND a.id = ?"
            params.append(analysis_id)
        query += " ORDER BY p.created_at, p.id"
        rows = database.get_db().execute(query, params).fetchall()
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(("record_id", "prediction", "category", "confidence", "timestamp"))
        writer.writerows(
            (row["id"], row["predicted_label"], row["attack_category"],
             f"{row['confidence']:.6f}", row["created_at"])
            for row in rows
        )
        return Response(
            stream.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=ids_report.csv"},
        )

    @app.get("/about")
    @login_required
    def about():
        return render_template("about.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

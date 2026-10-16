# Sentinel IDS

A local educational network intrusion detection application. It offers protected user accounts, NSL-KDD model training, CSV and manual predictions, private SQLite history, and exportable CSV reports. It does not capture live traffic or generate attacks.

## Run locally

Use Python 3.12 with the supplied environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000> and register an account. Set `IDS_SECRET_KEY` to a long random value before use beyond local development.

## Dataset and training

The original UNB/Canadian Institute for Cybersecurity page notes that NSL-KDD is no longer available from its site. A public GitHub mirror was used for this run. Download its headerless file to `datasets/KDDTrain+.txt`:

```powershell
New-Item -ItemType Directory -Force datasets | Out-Null
curl.exe --fail --location --output datasets\KDDTrain+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
```

The downloaded file used for the current model was 19,109,424 bytes, SHA-256 `1B86D2F957B33082081BBA410FE129B475EFEBCC13C9014C3F447C8271AADF95`; it contains 125,973 rows and 43 columns (41 traffic features, an attack label, and difficulty). Confirm the dataset source and hash before relying on results.

Train the binary Normal/Attack classifier:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Optional modes:

```powershell
.\.venv\Scripts\python.exe train_model.py --mode multiclass
.\.venv\Scripts\python.exe train_model.py --limit 10000 --epochs 5
```

The optional `--limit` is for quick local checks, not final evaluation. Training uses a reproducible stratified held-out test split. All reported evaluation metrics are computed from that held-out data; the script persists the ANN, fitted preprocessing and label objects, RF/LR baselines, feature schema, and metrics in `saved_models/`. Prediction routes stay unavailable until the trained ANN artifacts exist.

The ANN uses dense ReLU layers and dropout; preprocessing imputes missing numeric/categorical values, one-hot encodes categories and scales numeric features. Unknown categories at prediction time are ignored by the fitted encoder.

## Testing

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite checks route protection, registration/login/password hashing, missing-model behavior, safe CSV upload and report ownership, database persistence, input validation, and NSL-KDD label handling.

## Limitations

This is an academic demonstration, not a production security control. NSL-KDD is dated and results depend on the selected dataset. False positives and false negatives are expected. CSV uploads are parsed as data and deleted after processing. Use only authorized data.

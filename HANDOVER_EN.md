# Project Handover Document — Email Classification System (Welodge / Ollama)

**Written by:** Amir  
**Date:** May 18, 2026  
**For:** Next developer taking over this project

---

## 1. What This Project Does

This is an **automatic email/report classification system** for a disability support organization.  
Incoming reports (emails) written in Italian are automatically assigned to one of **8 social categories** related to disability rights. The system exposes a REST API that third-party systems can call to classify a report or submit corrective feedback.

The LLM used for classification is **qwen2.5:7b**, running locally through **Ollama** (no GPU required, no external API calls, fully self-hosted).

---

## 2. The 8 Classification Categories

| # | Category (Italian label) | Meaning |
|---|---|---|
| 1 | Accessibilità / barriere architettoniche / mobilità e trasporti / barriere digitali e media | Physical, digital or transport barriers |
| 2 | Altro | Generic / unclassifiable reports |
| 3 | Inclusione lavorativa | Job search, internships, mandatory placement (L.68/99) |
| 4 | Istruzione / formazione / inclusione scolastica | School support, PEI, DSA, inclusion |
| 5 | Rapporti con datori di lavoro | Conflicts with existing employers, L.104 leave |
| 6 | Salute / sanità / progetto di vita / assistenza domiciliare | Home care, prescriptions, life projects |
| 7 | Strutture socio-sanitarie | Issues inside residential/day-care facilities |
| 8 | Vita sociale / eventi / sport | Social exclusion, leisure, cultural events |

The label list is stored in `outputs/label_mapping.csv` and is the single source of truth at runtime.

---

## 3. Project Architecture

```
Incoming HTTP request
        │
        ▼
  FastAPI (app.py)           ← port 8003, Bearer token auth
        │
        ├─ POST /predict      ← classify an email → returns predicted label
        ├─ POST /feedback     ← submit correction if model was wrong
        └─ GET  /health       ← liveness check
        │
        ▼
  predict_email.py
        │
        ├─ Receives the text to classify in the `input` field
        ├─ Splits text into token chunks (max 1800 tokens each, using tiktoken cl100k_base)
        ├─ Sends each chunk to Ollama (qwen2.5:7b) with a structured Italian prompt
        └─ Majority-votes across chunk predictions → returns final label
        │
        ▼
  Ollama container (docker-compose)
        └─ Runs qwen2.5:7b locally on CPU
```

---

## 4. File-by-File Reference

| File | Purpose |
|---|---|
| `app.py` | FastAPI application — defines `/predict`, `/feedback`, `/health` endpoints |
| `predict_email.py` | Core inference logic — tokenization, chunking, Ollama calls, majority vote |
| `train_model.py` | **Legacy** — trains an Italian BERT model (`dbmdz/bert-base-italian-xxl-cased`). Not used in production any more (see § 9) |
| `retrain_with_feedback.py` | Fine-tunes the BERT model using corrected feedback from `feedback.csv` — also **legacy**, BERT is not used in inference |
| `evaluate_model_metrics.py` | Evaluates Ollama classification quality on a held-out split of `data/emails.csv` and writes results to `model_metrics.json` |
| `generate_emails.py` | Generates synthetic training emails (300 per category) and saves to `data/emails.csv` and `data/emails_full.json` |
| `issues_data.py` | Raw text templates used by `generate_emails.py` to build synthetic emails |
| `initialize_feedback_csv.py` | One-time script to create an empty `feedback.csv` with the correct columns |
| `schedule_retrain.py` | APScheduler script — meant to trigger retraining on the 1st of every month (has a known bug — see § 9) |
| `feedback.csv` | Accumulates rows where the model was wrong; used as retraining data |
| `model_metrics.json` | Latest evaluation metrics (accuracy, precision, recall, F1) |
| `data/emails.csv` | Training/evaluation dataset — columns: `text`, `label` |
| `data/emails_full.json` | Same dataset in JSON format with full email structure |
| `outputs/label_mapping.csv` | Ordered list of category labels — **do not change column order** |
| `dockerfile` | Docker image for the FastAPI app (Python 3.11-slim) |
| `docker-compose.yml` | Orchestrates 3 services: `ollama`, `ollama-init` (pulls the model once), `app` |
| `entrypoint.sh` | Shell script run by `ollama-init` — pulls qwen2.5:7b if not already cached |
| `requirements.txt` | Python dependencies |
| `.env` (not committed) | Must contain `API_BEARER_TOKEN` |

---

## 5. How to Run — Local (without Docker)

### Prerequisites
- Python 3.11+
- Ollama installed and running (`ollama serve`)
- qwen2.5:7b pulled: `ollama pull qwen2.5:7b`

### Steps

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate         # Linux/Mac
# venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create the .env file
echo "API_BEARER_TOKEN=your_secret_token_here" > .env

# 4. Initialize feedback file (only needed once)
python initialize_feedback_csv.py

# 5. Start the API
python app.py
# → API available at http://localhost:8003
```

---

## 6. How to Run — Docker (recommended for production)

```bash
# 1. Create .env file with your token
echo "API_BEARER_TOKEN=your_secret_token_here" > .env

# 2. Start all services
docker-compose up -d

# On first start, ollama-init will automatically pull qwen2.5:7b (~4 GB download)
# The app will only start after Ollama is healthy and the model is ready

# 3. Check logs
docker-compose logs -f app
docker-compose logs -f ollama-init

# 4. Stop everything
docker-compose down
```

The API will be available at `http://localhost:8003`.

---

## 7. API Usage

### Authentication
Every request must include:
```
Authorization: Bearer <your_token>
```

### POST /predict
Classify an email.

**Request body:**
```json
{
  "input": "Indicazioni Di seguito le indicazioni richieste: test di email"
}
```

**Response:**
```json
{
  "status": 200,
  "scores": [
    { "key": "Istruzione/formazione/inclusione scolastica", "value": "0,812" },
    { "key": "Altro", "value": "0,188" }
  ]
}
```

### POST /feedback
Submit a correction when the model predicted the wrong label.

**Request body:**
```json
{
  "input": "...",
  "correct_label": "Rapporti con datori di lavoro"
}
```
The correction is saved to `feedback.csv` only if the model's prediction differed from `correct_label`.

### GET /health
Returns `{"status": "ok"}` — used by Docker health checks and load balancers.

---

## 8. Feedback & Retraining Flow

```
User submits /feedback
        │
        ▼
  feedback.csv grows over time
        │
        ▼
  retrain_with_feedback.py   ← run manually or via schedule_retrain.py
        │  (fine-tunes BERT on feedback data)
        ▼
  outputs/model/             ← updated BERT weights
```

> **Important:** The current production inference path (`predict_email.py`) uses **Ollama, not BERT**. The retraining pipeline updates a BERT model that is currently **not connected to production**. See § 9 for details and the recommended fix.

---

## 9. Known Issues & Technical Debt

### 9.1 BERT vs Ollama split (most important)
The project started with a BERT classifier (`train_model.py`) and later migrated to Ollama (`predict_email.py`). However:
- `retrain_with_feedback.py` still fine-tunes BERT, not Ollama
- The BERT model weights in `outputs/model/` are **never used** during inference
- The feedback loop therefore has **no effect on production accuracy**

**Recommended fix:** Either:
- (A) Make the API serve from the BERT model (faster inference, no Ollama needed), or
- (B) Drop BERT entirely, keep Ollama, and implement feedback by periodically adjusting the prompt or switching to a fine-tunable open model

### 9.2 Hardcoded Windows path in `schedule_retrain.py`
```python
os.system("python e:\\amir\\classification_welodge\\retrain_with_feedback.py")
```
This path is from the developer's personal Windows machine. It will fail on any other system. Replace with a relative or dynamic path.

### 9.3 Stray syntax error in `predict_email.py`
At the end of the `CATEGORY_DESCRIPTIONS` list there is a stray string `siksjhah` that was accidentally left in. This causes a `SyntaxError` at import time.  
**Fix:** Remove the `siksjhah` line.

### 9.4 Stray file `=1.25.2` in root
A file named `=1.25.2` exists at the project root. It appears to be the result of a mistyped `pip install` command (`pip install package=1.25.2` instead of `==1.25.2`). It is safe to delete.

### 9.5 No `.env.example` file
There is no `.env.example` to guide setup. New developers must discover `API_BEARER_TOKEN` from reading the code.

### 9.6 Single Bearer token authentication
The API uses one static token for all callers. There is no per-client token management, no rotation, and no revocation mechanism.

### 9.7 No logging
There is no structured logging. Errors and predictions are not persisted anywhere beyond what Docker captures on stdout.

### 9.8 `schedule_retrain.py` uses `os.system()`
This is fragile. Use `subprocess.run()` or a proper task queue instead.

---

## 10. Suggested Future Improvements

- **Connect the feedback loop to production** (see § 9.1 above — highest priority)
- **Add a `.env.example`** so new developers can set up the environment quickly
- **Add unit tests** — at minimum for `predict_email.py` (chunking, sanitize_prediction) and the API endpoints
- **Structured logging** — use Python's `logging` module or a library like `structlog`; write logs to a file or a log aggregator
- **Multi-token API auth** — issue per-client tokens so individual clients can be revoked
- **Confidence scoring** — the model currently returns only a label; returning a confidence score would help downstream systems decide whether to escalate to a human reviewer
- **Active learning** — prioritize sending low-confidence predictions to human reviewers, then feed their corrections back automatically
- **Real email data** — the training dataset (`data/emails.csv`) is 100% synthetic. Replacing or augmenting it with real anonymized emails would significantly improve accuracy
- **Rate limiting** — the API has no rate limiting; add it via a reverse proxy (nginx, Traefik) or a FastAPI middleware
- **CI/CD pipeline** — add GitHub Actions (or equivalent) to run tests and build the Docker image on each push
- **Model versioning** — track which model version produced which prediction so you can roll back if a retrain degrades accuracy

---

## 11. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BEARER_TOKEN` | Yes | Secret token that API callers must include in the `Authorization` header |
| `OLLAMA_HOST` | No (defaults to `http://localhost:11434`) | Overridden in docker-compose to point to the `ollama` container |

---

## 12. Dependencies Summary

| Package | Role |
|---|---|
| `fastapi` + `uvicorn` | REST API server |
| `ollama` | Python client for the local Ollama server |
| `tiktoken` | Token counting (uses cl100k_base encoding to split long texts) |
| `pandas` | CSV I/O for feedback and label mapping |
| `apscheduler` | Cron-style scheduler for monthly retraining |
| `python-dotenv` | Loads `.env` into environment variables |
| `transformers` + `torch` | BERT training/retraining (legacy, not used in inference) |
| `scikit-learn` | Label encoding, metrics |
| `faker` | Synthetic data generation |

---

## 13. Contact

For any questions about design decisions made during development, feel free to reach out to the previous developer through the organization's internal channels.

Good luck with the project!

import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from predict_email import check_ollama_ready, predict

logger = logging.getLogger("uvicorn.error")

load_dotenv()
app = FastAPI()
API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN")

if not API_BEARER_TOKEN:
    raise RuntimeError("API_BEARER_TOKEN environment variable is required")

@app.get("/health")
def health_check():
    if not check_ollama_ready():
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "detail": "Ollama non disponibile o modello non caricato."},
        )
    return {"status": "ok"}

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Formato di autorizzazione non valido.")
    token = authorization[len("Bearer "):]
    if token != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Non autorizzato.")

class PredictRequest(BaseModel):
    input: str

@app.post("/predict")
def predict_route(request: PredictRequest, authorization: str = Header(...)):
    verify_token(authorization)
    try:
        return predict(request.model_dump())
    except Exception:
        logger.exception("Prediction failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": 503,
                "scores": [],
                "detail": "Servizio temporaneamente non disponibile. Riprovare tra qualche istante.",
            },
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=False)

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from predict_email import predict
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()
app = FastAPI()
API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN")

@app.get("/health")
def health_check():
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
    return predict(request.model_dump())

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=False)

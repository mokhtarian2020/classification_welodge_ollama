from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from predict_email import predict, json_to_bert_input
from dotenv import load_dotenv
import pandas as pd
import os
import uvicorn

load_dotenv()
app = FastAPI()
FEEDBACK_FILE = "feedback.csv"
API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN")

@app.get("/health")
def health_check():
    return {"status": "ok"}

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format.")
    token = authorization[len("Bearer "):]
    if token != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized.")

class EmailRequest(BaseModel):
    soggetto: str
    corpo: str
    allegati: list = []
    correct_label: str | None = None  # Opzionale

@app.post("/predict")
def predict_route(email: EmailRequest, authorization: str = Header(...)):
    verify_token(authorization)
    email_dict = email.dict(exclude_unset=True)
    label = predict(email_dict)
    return {
        "etichetta_predetta": label,
        "testo_input": json_to_bert_input(email_dict)
    }

@app.post("/feedback")
def feedback_route(email: EmailRequest, authorization: str = Header(...)):
    verify_token(authorization)
    if not email.correct_label:
        return {"errore": "Manca il campo correct_label."}

    email_dict = email.dict(exclude_unset=True)
    text = json_to_bert_input(email_dict)
    model_prediction = predict(email_dict)

    if model_prediction != email.correct_label:
        feedback_df = pd.DataFrame([{
            "Testo": text,
            "Predizione_Modello": model_prediction,
            "Etichetta_Corretta": email.correct_label
        }])

        if os.path.exists(FEEDBACK_FILE):
            feedback_df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
        else:
            feedback_df.to_csv(FEEDBACK_FILE, index=False)

        return {"messaggio": "Feedback registrato (la predizione del modello era diversa dall'etichetta corretta)."}
    else:
        return {"messaggio": "Nessun feedback registrato (la predizione del modello coincide con l'etichetta corretta)."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=False)

import pandas as pd
import os

FEEDBACK_FILE = "feedback.csv"

def initialize_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        pd.DataFrame(columns=["Testo", "Predizione_Modello", "Etichetta_Corretta"]).to_csv(FEEDBACK_FILE, index=False)
        print("✅ feedback.csv creato.")
    else:
        print("ℹ️ feedback.csv esiste già.")

if __name__ == "__main__":
    initialize_feedback_file()
  
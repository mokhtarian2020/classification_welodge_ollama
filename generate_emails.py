import pandas as pd
import random
import json
from pathlib import Path
import os
from faker import Faker

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
NUM_EMAILS_PER_CATEGORY = 300
OUTPUT_CSV              = "data/emails.csv"
OUTPUT_JSON             = "data/emails_full.json"
LABEL_MAPPING_PATH      = "outputs/label_mapping.csv"
fake = Faker("it_IT")

# --------------------------------------------------------------------------
# CATEGORIES & SUBJECT LINES
# --------------------------------------------------------------------------
CATEGORIES = [
    "Legale", "Contabilità", "Fatturazione", "Auto Aziendali",
    "Customer Relation", "Risorse Umane", "Acquisti", "Servizi"
]

subjects = {
    "Legale":           ["Consulenza legale urgente", "Clausola contrattuale dubbia", "Procedura legale"],
    "Contabilità":      ["Errore quadratura contabile", "Report spese incompleto", "Bilancio da rivedere"],
    "Fatturazione":     ["Fattura mancante", "Nota di credito errata", "Doppia fatturazione"],
    "Auto Aziendali":   ["Guasto veicolo aziendale", "Revisione auto", "Richiesta auto sostitutiva"],
    "Customer Relation":["Reclamo cliente", "Feedback negativo", "Ordine incompleto"],
    "Risorse Umane":    ["Richiesta ferie", "Errore cedolino", "Aggiornamento benefit"],
    "Acquisti":         ["Ordine in ritardo", "Preventivo urgente", "Fornitura non conforme"],
    "Servizi":          ["Guasto impianto", "Supporto tecnico", "Interruzione servizio"]
}

# --------------------------------------------------------------------------
# IMPORT ISSUES FROM EXTERNAL MODULE
# --------------------------------------------------------------------------
from issues_data import issues

# --------------------------------------------------------------------------
def generate_mock_email(category: str) -> dict:
    """Return one minimal, label-focused e-mail (subject + body)."""
    return {
        "soggetto": random.choice(subjects[category]),
        "corpo":    random.choice(issues[category]),
        "label":    category
    }

def load_label_mapping(path: str = LABEL_MAPPING_PATH):
    return pd.read_csv(path).squeeze().tolist() if os.path.exists(path) else None

# --------------------------------------------------------------------------
def generate_all_emails():
    emails = [
        generate_mock_email(cat)
        for cat in CATEGORIES
        for _   in range(NUM_EMAILS_PER_CATEGORY)
    ]

    # CSV for training
    df = pd.DataFrame([{
        "text":  f"{e['soggetto']} [SEP] {e['corpo']}",
        "label": e["label"]
    } for e in emails])

    mapping = load_label_mapping()
    if mapping:
        df = df[df["label"].isin(mapping)]
        df["label"] = pd.Categorical(df["label"], categories=mapping)

    Path("data").mkdir(exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")  # Specify encoding here

    # Full JSON (optional)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(emails, f, ensure_ascii=False, indent=2)

    print(f"✅  Generated {len(df)} e-mails → {OUTPUT_CSV}")
    print(f"📄  Full JSON saved → {OUTPUT_JSON}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    generate_all_emails()

import ollama
import pandas as pd
import tiktoken
from collections import Counter, defaultdict

# Constants
MODEL_NAME = "llama3.1"
CHUNK_SIZE_TOKENS = 1800
MAX_TOTAL_TOKENS = 2048
TOKEN_BUFFER = 200

# Load labels
LABELS = pd.read_csv("outputs/label_mapping.csv").squeeze().tolist()

# Use LLaMA-compatible tokenizer
ENCODING = tiktoken.get_encoding("cl100k_base")

def json_to_bert_input(email_json):
    soggetto = email_json.get("soggetto", "").strip()
    corpo = email_json.get("corpo", "").strip()
    allegati_list = email_json.get("allegati", [])

    allegati_text = " ".join(
        att.get("testo", "").strip()
        for att in allegati_list if isinstance(att, dict)
    ).strip()

    return f"{soggetto} [SEP] {corpo} [SEP] {allegati_text}"

def count_tokens(text):
    return len(ENCODING.encode(text))

def split_into_chunks(text, max_tokens=CHUNK_SIZE_TOKENS):
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        current_chunk.append(word)
        joined = " ".join(current_chunk)
        if count_tokens(joined) >= max_tokens:
            chunks.append(joined)
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def sanitize_prediction(pred: str) -> str:
    pred_clean = pred.strip().replace("*", "")
    if pred_clean in LABELS:
        return pred_clean
    raise ValueError(f"❌ Invalid prediction received: '{pred_clean}'")

def classify_chunk_ollama(chunk_text):
    prompt = f"""
Agisci come un esperto classificatore aziendale. Il tuo compito è leggere il contenuto di una email e rispondere **solo** con il nome esatto **di una singola categoria**, tra quelle elencate di seguito.

👉 Categorie disponibili: {', '.join(LABELS)}

⚠️ Non fornire spiegazioni, note, commenti o testo extra. Scrivi solo il nome della categoria.

Ora classifica la seguente email:
{chunk_text}

Categoria:
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Sei un classificatore aziendale. Rispondi solo con il nome della categoria senza testo extra."
                },
                {"role": "user", "content": prompt.strip()},
            ],
        )

        result = response["message"]["content"].strip()
        return sanitize_prediction(result)

    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")

def predict(email_json):
    full_text = json_to_bert_input(email_json)

    if count_tokens(full_text) <= (MAX_TOTAL_TOKENS - TOKEN_BUFFER):
        return classify_chunk_ollama(full_text)

    chunks = split_into_chunks(full_text)
    preds = []
    probs_sum = defaultdict(float)

    for chunk in chunks:
        label = classify_chunk_ollama(chunk)
        label_id = LABELS.index(label)
        preds.append(label_id)
        probs_sum[label_id] += 1

    counts = Counter(preds)
    most_common = counts.most_common()

    if len(chunks) == 2:
        a, b = preds[0], preds[1]
        return LABELS[a] if probs_sum[a] >= probs_sum[b] else LABELS[b]
    else:
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tied = [most_common[0][0], most_common[1][0]]
            best = max(tied, key=lambda x: probs_sum[x])
            return LABELS[best]
        else:
            return LABELS[most_common[0][0]]

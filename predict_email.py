import ollama
import pandas as pd
import tiktoken
from collections import Counter, defaultdict

# Constants
MODEL_NAME = "qwen2.5:7b"
CHUNK_SIZE_TOKENS = 1800
MAX_TOTAL_TOKENS = 2048
TOKEN_BUFFER = 200

# Load labels
LABELS = pd.read_csv("outputs/label_mapping.csv").squeeze().tolist()

NUMBERED_LABELS = {str(i + 1): label for i, label in enumerate(LABELS)}

CATEGORY_DESCRIPTIONS = [
    "la segnalazione riguarda PRINCIPALMENTE una barriera fisica, digitale o di trasporto in sé (rampa rotta, sito web inaccessibile, bus senza pedana, semaforo senza segnale sonoro) — NON usare se il focus è l'esclusione sociale",
    "segnalazioni generiche, richieste di orientamento, casi non classificabili o che toccano più categorie",
    "RICERCA di lavoro: tirocini, borse lavoro, collocamento obbligatorio L.68/99, job coach, percorsi di inserimento — NON usare se c'è già un rapporto di lavoro attivo",
    "sostegno scolastico, insegnante di sostegno, PEI, PDP, DSA, inclusione in classe, difficoltà scolastiche",
    "conflitti o problemi con un DATORE DI LAVORO esistente: discriminazione, permessi L.104, licenziamento, adattamento postazione — NON usare per chi cerca lavoro",
    "cure e assistenza ricevute A DOMICILIO o in modo individuale: ADI, ausili prescritti, progetto di vita personale, caregiver familiare, invalidità civile — NON riguarda strutture residenziali",
    "problemi con il FUNZIONAMENTO di una struttura collettiva dove la persona vive o frequenta: RSA, centro diurno, casa famiglia, comunità — il problema è nella struttura, NON nelle cure a casa",
    "ESCLUSIONE o ISOLAMENTO dalla vita sociale: teatro, eventi culturali, sport amatoriale, attività ricreative, vacanze, aggregazione — usare quando il focus è la partecipazione sociale negata, anche se causa è una barriera",
]

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
    # Extract first token — model should respond with just a digit
    token = pred.strip().split()[0].rstrip(".").strip() if pred.strip() else ""
    if token in NUMBERED_LABELS:
        return NUMBERED_LABELS[token]
    # Fallback: exact name match
    if token in LABELS:
        return token
    raise ValueError(f"❌ Invalid prediction received: '{pred.strip()}'")

def classify_chunk_ollama(chunk_text):
    numbered = "\n".join(
        f"{i+1}. {label} — {CATEGORY_DESCRIPTIONS[i]}"
        for i, label in enumerate(LABELS)
    )
    prompt = f"""Sei un esperto classificatore di segnalazioni sociali per persone con disabilità.
Leggi il testo seguente e rispondi SOLO con il numero della categoria corretta.

Categorie:
{numbered}

REGOLE DI DISAMBIGUAZIONE (leggi con attenzione):
- Se il testo parla di ESCLUSIONE da eventi, sport, tempo libero o vita sociale → scegli 8, anche se menziona barriere fisiche
- Se il testo parla di una BARRIERA FISICA/DIGITALE come problema principale (rampa, sito, bus) senza contesto sociale → scegli 1
- Se il problema riguarda una STRUTTURA RESIDENZIALE o centro diurno (personale, regole, condizioni interne) → scegli 7
- Se il problema riguarda CURE O ASSISTENZA A DOMICILIO o ausili individuali → scegli 6

Regole di formato:
- Rispondi ESCLUSIVAMENTE con un numero intero da 1 a {len(LABELS)}.
- Nessuna spiegazione, nessun testo aggiuntivo.

Testo da classificare:
{chunk_text}

Numero categoria:"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            options={"temperature": 0},
            messages=[
                {
                    "role": "system",
                    "content": f"Sei un classificatore di segnalazioni sociali per persone con disabilità. Rispondi SOLO con un numero intero da 1 a {len(LABELS)}. Nessun testo aggiuntivo."
                },
                {"role": "user", "content": prompt},
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

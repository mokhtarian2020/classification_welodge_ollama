import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache

import tiktoken

# Constants
MODEL_NAME = "qwen2.5:3b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHUNK_SIZE_TOKENS = 1800
MAX_TOTAL_TOKENS = 2048
TOKEN_BUFFER = 200
PREDICT_CACHE_SIZE = int(os.environ.get("PREDICT_CACHE_SIZE", "512"))
OLLAMA_NUM_CTX = 2048

ALTRO_KEY = "Altro"

CATEGORIES = [
    {
        "key": "Istruzione, formazione e inclusione socio-lavorativa (ambito tematico 01)",
        "description": (
            "scuola, PEI, PDP, insegnante di sostegno, inclusione scolastica, formazione professionale, "
            "ricerca lavoro, tirocini, borse lavoro, collocamento mirato, inserimento lavorativo, "
            "licenziamento, discriminazione sul lavoro, permessi legge 104, rapporti con datori di lavoro"
        ),
    },
    {
        "key": "Servizi sociosanitari, progetto di vita e assistenza (ambito tematico 02)",
        "description": (
            "RSA, centri diurni, comunità alloggio, strutture residenziali, assistenza domiciliare ADI, "
            "ausili sanitari, cure mediche, invalidità civile, progetto di vita individuale, "
            "assistenza e cura della persona"
        ),
    },
    {
        "key": "Accessibilità, mobilità e tecnologie inclusive (ambito tematico 03)",
        "description": (
            "barriere architettoniche e digitali: rampe, ascensori, parcheggi disabili, bus, treni, "
            "trasporto pubblico, mobilità, siti web non accessibili, screen reader, moduli online, "
            "app, media digitali. Include ogni problema tecnico di accessibilità fisica o informatica"
        ),
    },
    {
        "key": "Partecipazione sociale, culturale, ricreativa e sportiva (ambito tematico 04)",
        "description": (
            "partecipazione ed esclusione da attività sociali: teatro, cinema, musei, eventi culturali, "
            "sport, palestre, corsi ricreativi, vacanze, turismo, associazionismo, feste e manifestazioni, "
            "accesso e fruizione di servizi e iniziative aperti al pubblico"
        ),
    },
]

# The model answers with a single digit: 1-4 = thematic area, 0 = Altro.
DIGIT_TO_KEY = {"0": ALTRO_KEY}
DIGIT_TO_KEY.update({str(i + 1): category["key"] for i, category in enumerate(CATEGORIES)})

ALL_CATEGORY_KEYS = [category["key"] for category in CATEGORIES] + [ALTRO_KEY]

CATEGORY_OPTIONS = "\n".join(
    f"{i + 1} = {category['key']}: {category['description']}"
    for i, category in enumerate(CATEGORIES)
)

# Deterministic safety net: unambiguous employment-relationship terms force
# area 01 to the top. Prompt tuning for these cases proved fragile (fixing one
# example regressed others), so this narrow rule is applied after the model.
EMPLOYMENT_CATEGORY_KEY = CATEGORIES[0]["key"]
EMPLOYMENT_PATTERN = re.compile(
    r"licenzi|mobbing|demansion|datore\s+di\s+lavoro|discriminazione\s+sul\s+lavoro",
    re.IGNORECASE,
)
EMPLOYMENT_BOOST = 1.5

ENCODING = tiktoken.get_encoding("cl100k_base")


def extract_input(request_json):
    return request_json.get("input", "").strip()


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


def format_probability(probability):
    if probability < 0.0005:
        return "0,0"
    return f"{probability:.3f}".replace(".", ",")


def _ollama_chat(payload, retries=3):
    request = urllib.request.Request(
        f"{OLLAMA_HOST.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            raise RuntimeError(f"Errore API Ollama ({exc.code}): {body}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Errore API Ollama: {last_error}") from last_error


def check_ollama_ready():
    """Return True if Ollama is reachable and the model is loaded."""
    try:
        tags_request = urllib.request.Request(f"{OLLAMA_HOST.rstrip('/')}/api/tags")
        with urllib.request.urlopen(tags_request, timeout=5) as response:
            tags = json.loads(response.read())
        models = [model.get("name", "") for model in tags.get("models", [])]
        return any(MODEL_NAME in name for name in models)
    except Exception:
        return False


def _uniform_probabilities():
    return {key: 1.0 / len(ALL_CATEGORY_KEYS) for key in ALL_CATEGORY_KEYS}


def _digit_probabilities(logprob_entry):
    """Turn the logprobs of the answer digit into a probability per category."""
    digit_logprobs = {}

    for item in [logprob_entry] + logprob_entry.get("top_logprobs", []):
        token = item.get("token", "").strip()
        if token in DIGIT_TO_KEY:
            digit_logprobs[token] = max(
                digit_logprobs.get(token, float("-inf")),
                item.get("logprob", float("-inf")),
            )

    if not digit_logprobs:
        return None

    max_logprob = max(digit_logprobs.values())
    weights = {
        digit: math.exp(logprob - max_logprob)
        for digit, logprob in digit_logprobs.items()
    }
    total = sum(weights.values())

    probabilities = {key: 0.0 for key in ALL_CATEGORY_KEYS}
    for digit, weight in weights.items():
        probabilities[DIGIT_TO_KEY[digit]] = weight / total

    return probabilities


def _classify_text(text):
    prompt = (
        "Sei un classificatore di segnalazioni per persone con disabilità.\n"
        "Scegli la categoria che descrive meglio il testo.\n"
        f"{CATEGORY_OPTIONS}\n"
        "0 = Altro: saluti, richieste generiche di informazioni, test, "
        "testi senza una segnalazione specifica o non attribuibili alle categorie sopra\n"
        f"Testo: {text}\n"
        "Rispondi SOLO con il numero della categoria (0, 1, 2, 3 o 4)."
    )

    response = _ollama_chat(
        {
            "model": MODEL_NAME,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "messages": [
                {"role": "system", "content": "Rispondi solo con un numero: 0, 1, 2, 3 o 4."},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0,
                "num_predict": 1,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        }
    )

    logprob_entries = response.get("logprobs") or []
    if not logprob_entries:
        return _uniform_probabilities()

    probabilities = _digit_probabilities(logprob_entries[0])
    return probabilities if probabilities is not None else _uniform_probabilities()


def _merge_probability_maps(probability_maps):
    merged = {key: 0.0 for key in ALL_CATEGORY_KEYS}
    for probability_map in probability_maps:
        for key, probability in probability_map.items():
            merged[key] += probability

    count = len(probability_maps) or 1
    return {key: value / count for key, value in merged.items()}


def build_scores_response(probabilities):
    scores = [
        {"key": key, "value": format_probability(probability)}
        for key, probability in probabilities.items()
    ]
    scores.sort(key=lambda item: float(item["value"].replace(",", ".")), reverse=True)
    return {"status": 200, "scores": scores}


def _apply_employment_rule(text, probabilities):
    if not EMPLOYMENT_PATTERN.search(text):
        return probabilities

    top_key = max(probabilities, key=probabilities.get)
    if top_key == EMPLOYMENT_CATEGORY_KEY:
        return probabilities

    adjusted = dict(probabilities)
    adjusted[EMPLOYMENT_CATEGORY_KEY] = probabilities[top_key] * EMPLOYMENT_BOOST
    total = sum(adjusted.values())
    return {key: value / total for key, value in adjusted.items()}


def _predict_uncached(full_text):
    if count_tokens(full_text) <= (MAX_TOTAL_TOKENS - TOKEN_BUFFER):
        probabilities = _classify_text(full_text)
    else:
        chunks = split_into_chunks(full_text)
        probabilities = _merge_probability_maps([_classify_text(chunk) for chunk in chunks])

    probabilities = _apply_employment_rule(full_text, probabilities)
    return build_scores_response(probabilities)


@lru_cache(maxsize=PREDICT_CACHE_SIZE)
def _predict_cached(full_text):
    return _predict_uncached(full_text)


def predict(request_json):
    full_text = extract_input(request_json)
    if not full_text:
        return build_scores_response(_uniform_probabilities())

    cached = _predict_cached(full_text)
    return {
        "status": cached["status"],
        "scores": [{"key": score["key"], "value": score["value"]} for score in cached["scores"]],
    }

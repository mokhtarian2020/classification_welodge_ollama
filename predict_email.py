import json
import math
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import tiktoken

# Constants
MODEL_NAME = "qwen2.5:3b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHUNK_SIZE_TOKENS = 1800
MAX_TOTAL_TOKENS = 2048
TOKEN_BUFFER = 200
SCORE_SMOOTHING = 2.0
SCORE_PRIOR = 0.02
SCORE_PRIOR_ALTRO = 0.002
ALTRO_KEY = "Altro"
PREDICT_CACHE_SIZE = int(os.environ.get("PREDICT_CACHE_SIZE", "512"))
OLLAMA_RATING_NUM_CTX = 2048
OLLAMA_MAX_WORKERS = int(os.environ.get("OLLAMA_MAX_WORKERS", "4"))

CATEGORIES = [
    {
        "key": "Istruzione, formazione e inclusione socio-lavorativa (ambito tematico 01)",
        "description": (
            "istruzione, formazione professionale, inclusione scolastica, rapporti con il sistema "
            "educativo e con i datori di lavoro, inclusione lavorativa delle persone con disabilità"
        ),
    },
    {
        "key": "Servizi sociosanitari, progetto di vita e assistenza (ambito tematico 02)",
        "description": (
            "strutture sociosanitarie, progetto di vita individuale, assistenza domiciliare "
            "e servizi di supporto alla persona"
        ),
    },
    {
        "key": "Accessibilità, mobilità e tecnologie inclusive (ambito tematico 03)",
        "description": (
            "accessibilità fisica e digitale, eliminazione delle barriere architettoniche, "
            "mobilità e trasporti, accessibilità dei media e dei servizi digitali"
        ),
    },
    {
        "key": "Partecipazione sociale, culturale, ricreativa e sportiva (ambito tematico 04)",
        "description": (
            "attività sociali, culturali, ricreative e sportive; eventi e manifestazioni pubbliche "
            "e private; associazionismo; turismo accessibile; accesso e fruizione di servizi "
            "e iniziative aperti al pubblico"
        ),
    },
    {
        "key": ALTRO_KEY,
        "description": "solo segnalazioni generiche, richieste di orientamento o casi davvero non classificabili",
    },
]

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


def _ollama_chat(payload):
    request = urllib.request.Request(
        f"{OLLAMA_HOST.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        raise RuntimeError(f"Errore API Ollama ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Errore API Ollama: {exc}") from exc


def _parse_digit_rating(logprob_entry):
    digit_logprobs = {}

    for item in [logprob_entry] + logprob_entry.get("top_logprobs", []):
        token = item.get("token", "").strip()
        if re.fullmatch(r"[0-9]", token):
            digit_logprobs[token] = max(
                digit_logprobs.get(token, float("-inf")),
                item.get("logprob", float("-inf")),
            )

    if not digit_logprobs:
        return 0.0

    max_logprob = max(digit_logprobs.values())
    weighted_sum = sum(
        int(digit) * math.exp(logprob - max_logprob)
        for digit, logprob in digit_logprobs.items()
    )
    total = sum(math.exp(logprob - max_logprob) for logprob in digit_logprobs.values())
    return weighted_sum / total


def _category_rating(category, text):
    extra = ""
    if category["key"] == ALTRO_KEY:
        extra = (
            " Usa 9 SOLO se il testo non riguarda in alcun modo scuola, lavoro, salute, "
            "strutture, accessibilità, mobilità, vita sociale o eventi. "
            "Se anche solo parzialmente pertinente a un ambito tematico, rispondi 0."
        )

    prompt = (
        f"Sei un classificatore di segnalazioni per persone con disabilità.\n"
        f"Valuta la pertinenza del testo alla categoria (0=per nulla pertinente, 9=massima pertinenza).\n"
        f"Categoria: {category['key']}\n"
        f"Definizione: {category['description']}{extra}\n"
        f"Testo: {text}\n"
        f"Rispondi SOLO con un numero intero da 0 a 9."
    )

    response = _ollama_chat(
        {
            "model": MODEL_NAME,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "messages": [
                {"role": "system", "content": "Rispondi solo con un numero intero da 0 a 9."},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0,
                "num_predict": 1,
                "num_ctx": OLLAMA_RATING_NUM_CTX,
            },
        }
    )

    logprob_entries = response.get("logprobs") or []
    if not logprob_entries:
        return category["key"], 0.0

    return category["key"], _parse_digit_rating(logprob_entries[0])


def _ratings_for_text(text):
    ratings = {}
    max_workers = min(OLLAMA_MAX_WORKERS, len(CATEGORIES))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_category_rating, category, text) for category in CATEGORIES]
        for future in as_completed(futures):
            key, rating = future.result()
            ratings[key] = rating

    return ratings


def _category_prior(key):
    return SCORE_PRIOR_ALTRO if key == ALTRO_KEY else SCORE_PRIOR


def _ratings_to_probabilities(ratings):
    weights = {
        key: math.exp(rating / SCORE_SMOOTHING) + _category_prior(key)
        for key, rating in ratings.items()
    }
    total = sum(weights.values()) or 1.0
    return {key: weight / total for key, weight in weights.items()}


def _merge_rating_maps(rating_maps):
    merged = {category["key"]: 0.0 for category in CATEGORIES}
    for rating_map in rating_maps:
        for key, rating in rating_map.items():
            merged[key] += rating

    count = len(rating_maps) or 1
    return {key: value / count for key, value in merged.items()}


def build_scores_response(probabilities):
    scores = [
        {"key": key, "value": format_probability(probability)}
        for key, probability in probabilities.items()
    ]
    scores.sort(key=lambda item: float(item["value"].replace(",", ".")), reverse=True)
    return {"status": 200, "scores": scores}


def _predict_uncached(full_text):
    if count_tokens(full_text) <= (MAX_TOTAL_TOKENS - TOKEN_BUFFER):
        ratings = _ratings_for_text(full_text)
    else:
        chunks = split_into_chunks(full_text)
        chunk_ratings = [_ratings_for_text(chunk) for chunk in chunks]
        ratings = _merge_rating_maps(chunk_ratings)

    probabilities = _ratings_to_probabilities(ratings)
    return build_scores_response(probabilities)


@lru_cache(maxsize=PREDICT_CACHE_SIZE)
def _predict_cached(full_text):
    return _predict_uncached(full_text)


def predict(request_json):
    full_text = extract_input(request_json)
    if not full_text:
        probabilities = {category["key"]: 1.0 / len(CATEGORIES) for category in CATEGORIES}
        return build_scores_response(probabilities)

    cached = _predict_cached(full_text)
    return {
        "status": cached["status"],
        "scores": [{"key": score["key"], "value": score["value"]} for score in cached["scores"]],
    }

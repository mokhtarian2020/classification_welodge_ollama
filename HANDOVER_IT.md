# Documento di Passaggio di Consegne — Sistema di Classificazione Email (Welodge / Ollama)

**Scritto da:** Amir  
**Data:** 18 maggio 2026  
**Per:** Il prossimo sviluppatore che lavorerà su questo progetto

---

## 1. Cosa fa questo progetto

Si tratta di un **sistema automatico di classificazione di email/segnalazioni** per un'organizzazione di supporto alle persone con disabilità.  
Le segnalazioni in arrivo, scritte in italiano, vengono assegnate automaticamente a una delle **8 categorie sociali** legate ai diritti delle persone con disabilità. Il sistema espone un'API REST che sistemi terzi possono chiamare per classificare una segnalazione o inviare una correzione (feedback).

Il modello LLM usato per la classificazione è **qwen2.5:7b**, eseguito localmente tramite **Ollama** (non richiede GPU, nessuna chiamata a API esterne, completamente self-hosted).

---

## 2. Le 8 Categorie di Classificazione

| # | Etichetta | Significato |
|---|---|---|
| 1 | Accessibilità / barriere architettoniche / mobilità e trasporti / barriere digitali e media | Barriere fisiche, digitali o di trasporto |
| 2 | Altro | Segnalazioni generiche o non classificabili |
| 3 | Inclusione lavorativa | Ricerca di lavoro, tirocini, collocamento obbligatorio L.68/99 |
| 4 | Istruzione / formazione / inclusione scolastica | Sostegno scolastico, PEI, DSA, inclusione in classe |
| 5 | Rapporti con datori di lavoro | Conflitti con datori di lavoro esistenti, permessi L.104 |
| 6 | Salute / sanità / progetto di vita / assistenza domiciliare | Cure domiciliari, ausili, progetto di vita individuale |
| 7 | Strutture socio-sanitarie | Problemi interni a strutture residenziali o centri diurni |
| 8 | Vita sociale / eventi / sport | Esclusione sociale, attività ricreative, eventi culturali |

La lista delle etichette è salvata in `outputs/label_mapping.csv` ed è l'unica fonte di verità a runtime. **Non modificare l'ordine delle righe.**

---

## 3. Architettura del Progetto

```
Richiesta HTTP in arrivo
        │
        ▼
  FastAPI (app.py)           ← porta 8003, autenticazione Bearer token
        │
        ├─ POST /predict      ← classifica un'email → restituisce l'etichetta predetta
        ├─ POST /feedback     ← invia una correzione se il modello ha sbagliato
        └─ GET  /health       ← controllo di vita (liveness check)
        │
        ▼
  predict_email.py
        │
        ├─ Riceve il testo da classificare nel campo `input`
        ├─ Divide il testo in chunk di token (max 1800 token, usando tiktoken cl100k_base)
        ├─ Invia ogni chunk a Ollama (qwen2.5:7b) con un prompt strutturato in italiano
        └─ Voto di maggioranza tra le predizioni dei chunk → restituisce l'etichetta finale
        │
        ▼
  Container Ollama (docker-compose)
        └─ Esegue qwen2.5:7b in locale su CPU
```

---

## 4. Guida ai File del Progetto

| File | Scopo |
|---|---|
| `app.py` | Applicazione FastAPI — definisce gli endpoint `/predict`, `/feedback`, `/health` |
| `predict_email.py` | Logica principale di inferenza — tokenizzazione, chunking, chiamate Ollama, voto di maggioranza |
| `train_model.py` | **Legacy** — addestra un modello BERT italiano (`dbmdz/bert-base-italian-xxl-cased`). Non più usato in produzione (vedi § 9) |
| `retrain_with_feedback.py` | Fine-tuning del modello BERT usando le correzioni in `feedback.csv` — anch'esso **legacy**, BERT non è usato nell'inferenza |
| `evaluate_model_metrics.py` | Valuta la qualità della classificazione Ollama su una porzione di test di `data/emails.csv` e salva i risultati in `model_metrics.json` |
| `generate_emails.py` | Genera email di addestramento sintetiche (300 per categoria) e le salva in `data/emails.csv` e `data/emails_full.json` |
| `issues_data.py` | Testi template usati da `generate_emails.py` per costruire le email sintetiche |
| `initialize_feedback_csv.py` | Script da eseguire una sola volta per creare un `feedback.csv` vuoto con le colonne corrette |
| `schedule_retrain.py` | Script APScheduler — intende avviare il retraining il 1° di ogni mese (ha un bug noto — vedi § 9) |
| `feedback.csv` | Accumula le righe in cui il modello ha sbagliato; usato come dato di retraining |
| `model_metrics.json` | Ultime metriche di valutazione (accuracy, precision, recall, F1) |
| `data/emails.csv` | Dataset di addestramento/valutazione — colonne: `text`, `label` |
| `data/emails_full.json` | Stesso dataset in formato JSON con struttura email completa |
| `outputs/label_mapping.csv` | Lista ordinata delle etichette — **non modificare l'ordine delle righe** |
| `dockerfile` | Immagine Docker per l'app FastAPI (Python 3.11-slim) |
| `docker-compose.yml` | Orchestra 3 servizi: `ollama`, `ollama-init` (scarica il modello una sola volta), `app` |
| `entrypoint.sh` | Script shell eseguito da `ollama-init` — scarica qwen2.5:7b se non è già presente nella cache |
| `requirements.txt` | Dipendenze Python |
| `.env` (non committato) | Deve contenere `API_BEARER_TOKEN` |

---

## 5. Come Avviare il Progetto — In Locale (senza Docker)

### Prerequisiti
- Python 3.11+
- Ollama installato e in esecuzione (`ollama serve`)
- Modello qwen2.5:7b scaricato: `ollama pull qwen2.5:7b`

### Passaggi

```bash
# 1. Crea e attiva l'ambiente virtuale
python -m venv venv
source venv/bin/activate         # Linux/Mac
# venv\Scripts\activate          # Windows

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. Crea il file .env
echo "API_BEARER_TOKEN=il_tuo_token_segreto" > .env

# 4. Inizializza il file feedback (solo la prima volta)
python initialize_feedback_csv.py

# 5. Avvia l'API
python app.py
# → API disponibile su http://localhost:8003
```

---

## 6. Come Avviare il Progetto — Docker (raccomandato per la produzione)

```bash
# 1. Crea il file .env con il tuo token
echo "API_BEARER_TOKEN=il_tuo_token_segreto" > .env

# 2. Avvia tutti i servizi
docker-compose up -d

# Al primo avvio, ollama-init scaricherà automaticamente qwen2.5:7b (~4 GB)
# L'app si avvierà solo dopo che Ollama è sano e il modello è pronto

# 3. Controlla i log
docker-compose logs -f app
docker-compose logs -f ollama-init

# 4. Ferma tutto
docker-compose down
```

L'API sarà disponibile su `http://localhost:8003`.

---

## 7. Utilizzo dell'API

### Autenticazione
Ogni richiesta deve includere:
```
Authorization: Bearer <il_tuo_token>
```

### POST /predict
Classifica un'email.

**Corpo della richiesta:**
```json
{
  "input": "Indicazioni Di seguito le indicazioni richieste: test di email"
}
```

**Risposta:**
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
Invia una correzione quando il modello ha predetto l'etichetta sbagliata.

**Corpo della richiesta:**
```json
{
  "input": "...",
  "etichetta_corretta": "Rapporti con datori di lavoro"
}
```
La correzione viene salvata in `feedback.csv` solo se la predizione del modello differisce da `etichetta_corretta`.

### GET /health
Restituisce `{"status": "ok"}` — usato dai health check di Docker e dai load balancer.

---

## 8. Flusso di Feedback e Retraining

```
L'utente invia /feedback
        │
        ▼
  feedback.csv cresce nel tempo
        │
        ▼
  retrain_with_feedback.py   ← eseguire manualmente o tramite schedule_retrain.py
        │  (fine-tuning di BERT sui dati di feedback)
        ▼
  outputs/model/             ← pesi BERT aggiornati
```

> **Importante:** L'attuale percorso di inferenza in produzione (`predict_email.py`) usa **Ollama, non BERT**. Il pipeline di retraining aggiorna un modello BERT che attualmente **non è collegato alla produzione**. Vedi § 9 per i dettagli e la correzione raccomandata.

---

## 9. Problemi Noti e Debito Tecnico

### 9.1 Separazione BERT vs Ollama (priorità massima)
Il progetto è iniziato con un classificatore BERT (`train_model.py`) ed è poi migrato a Ollama (`predict_email.py`). Tuttavia:
- `retrain_with_feedback.py` esegue ancora il fine-tuning di BERT, non di Ollama
- I pesi del modello BERT in `outputs/model/` **non vengono mai usati** durante l'inferenza
- Il ciclo di feedback non ha quindi **alcun effetto sull'accuratezza in produzione**

**Correzione raccomandata:** Scegliere una delle due strade:
- (A) Far servire l'API dal modello BERT (inferenza più veloce, Ollama non necessario), oppure
- (B) Eliminare completamente BERT, mantenere Ollama, e implementare il feedback aggiustando il prompt periodicamente o migrando verso un modello open fine-tunable

### 9.2 Percorso Windows hardcoded in `schedule_retrain.py`
```python
os.system("python e:\\amir\\classification_welodge\\retrain_with_feedback.py")
```
Questo percorso è della macchina Windows personale dello sviluppatore. Fallirà su qualsiasi altro sistema. Sostituire con un percorso relativo o dinamico.

### 9.3 Errore di sintassi in `predict_email.py`
Alla fine della lista `CATEGORY_DESCRIPTIONS` c'è una stringa `siksjhah` lasciata per errore. Questo causa un `SyntaxError` all'importazione del modulo.  
**Correzione:** Rimuovere la riga con `siksjhah`.

### 9.4 File anomalo `=1.25.2` nella directory radice
Esiste un file chiamato `=1.25.2` nella root del progetto. Sembra il risultato di un comando `pip install` digitato male (`pip install pacchetto=1.25.2` invece di `==1.25.2`). È sicuro eliminarlo.

### 9.5 Mancanza del file `.env.example`
Non esiste un file `.env.example` per guidare la configurazione. I nuovi sviluppatori devono scoprire la variabile `API_BEARER_TOKEN` leggendo il codice.

### 9.6 Autenticazione con singolo Bearer token
L'API usa un unico token statico per tutti i chiamanti. Non esiste gestione per-client, rotazione o revoca dei token.

### 9.7 Nessun sistema di logging
Non è presente un logging strutturato. Errori e predizioni non vengono persistiti da nessuna parte al di là di ciò che Docker cattura su stdout.

### 9.8 `schedule_retrain.py` usa `os.system()`
Questo approccio è fragile. Usare `subprocess.run()` o una coda di task dedicata.

---

## 10. Miglioramenti Futuri Consigliati

- **Collegare il ciclo di feedback alla produzione** (vedi § 9.1 — priorità massima)
- **Aggiungere un file `.env.example`** per permettere ai nuovi sviluppatori di configurare l'ambiente rapidamente
- **Aggiungere unit test** — almeno per `predict_email.py` (chunking, `sanitize_prediction`) e per gli endpoint dell'API
- **Logging strutturato** — usare il modulo `logging` di Python o una libreria come `structlog`; scrivere i log su file o su un aggregatore di log
- **Autenticazione multi-token** — emettere token per-client così da poter revocare singoli client
- **Punteggi di confidenza** — l'endpoint `/predict` restituisce probabilità per tutte le categorie nel campo `scores`
- **Active learning** — prioritizzare le predizioni a bassa confidenza per la revisione umana, poi reintrodurre automaticamente le correzioni nel ciclo di feedback
- **Dati email reali** — il dataset di addestramento (`data/emails.csv`) è composto al 100% da dati sintetici. Sostituirlo o arricchirlo con email reali anonimizzate migliorerebbe significativamente l'accuratezza
- **Rate limiting** — l'API non ha limiti di frequenza; aggiungerne tramite un reverse proxy (nginx, Traefik) o un middleware FastAPI
- **Pipeline CI/CD** — aggiungere GitHub Actions (o equivalente) per eseguire i test e costruire l'immagine Docker ad ogni push
- **Versionamento del modello** — tracciare quale versione del modello ha prodotto quale predizione, così da poter fare rollback se un retraining peggiora l'accuratezza

---

## 11. Variabili d'Ambiente

| Variabile | Obbligatoria | Descrizione |
|---|---|---|
| `API_BEARER_TOKEN` | Sì | Token segreto che i chiamanti dell'API devono includere nell'header `Authorization` |
| `OLLAMA_HOST` | No (default: `http://localhost:11434`) | Sovrascritta in docker-compose per puntare al container `ollama` |

---

## 12. Riepilogo delle Dipendenze

| Pacchetto | Ruolo |
|---|---|
| `fastapi` + `uvicorn` | Server API REST |
| `ollama` | Client Python per il server Ollama locale |
| `tiktoken` | Conteggio token (usa la codifica cl100k_base per dividere i testi lunghi) |
| `pandas` | Lettura/scrittura CSV per feedback e label mapping |
| `apscheduler` | Scheduler cron per il retraining mensile |
| `python-dotenv` | Carica il file `.env` nelle variabili d'ambiente |
| `transformers` + `torch` | Training/retraining BERT (legacy, non usato nell'inferenza) |
| `scikit-learn` | Label encoding, metriche di valutazione |
| `faker` | Generazione di dati sintetici |

---

## 13. Contatti

Per qualsiasi domanda sulle scelte progettuali effettuate durante lo sviluppo, contattare il precedente sviluppatore tramite i canali interni dell'organizzazione.

In bocca al lupo con il progetto!

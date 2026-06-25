# Guida all'Integrazione API — Servizio di Classificazione Email

**Endpoint:** https://ceia.gesan.it

---

## Autenticazione

Tutte le richieste a `POST /predict` richiedono un token Bearer nell'intestazione `Authorization`:

```
Authorization: Bearer <token>
```

> **Mantenere il token riservato — non condividerlo pubblicamente.**

`GET /health` non richiede autenticazione.

---

## POST /predict

Classifica un testo valutando **tutte le 8 categorie** con un punteggio pesato. La somma degli `value` in `scores` è **1**.

### ESEMPIO INPUT

```json
{"input": "Indicazioni Di seguito le indicazioni richieste: test di email"}
```

| Campo   | Tipo   | Obbligatorio | Descrizione |
|---------|--------|--------------|-------------|
| `input` | string | sì           | Testo completo da classificare (oggetto, corpo e allegati già uniti dal client in un'unica stringa) |

### Esempio richiesta (curl)

```bash
curl -X POST https://ceia.gesan.it/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"input": "Indicazioni Di seguito le indicazioni richieste: test di email"}'
```

### ESEMPIO OUTPUT

```json
{
  "status": 200,
  "scores": [
    { "key": "Accessibilità/barriere architettoniche/mobilità e trasporti/barriere digitali e media", "value": "0,921" },
    { "key": "Vita sociale/eventi/sport", "value": "0,079" },
    { "key": "Altro", "value": "0,0" },
    { "key": "Inclusione lavorativa", "value": "0,0" },
    { "key": "Istruzione/formazione/inclusione scolastica", "value": "0,0" },
    { "key": "Rapporti con datori di lavoro", "value": "0,0" },
    { "key": "Salute/sanità/progetto di vita/assistenza domiciliare", "value": "0,0" },
    { "key": "Strutture socio-sanitarie", "value": "0,0" }
  ]
}
```

| Campo    | Tipo   | Descrizione |
|----------|--------|-------------|
| `status` | int    | Codice di esito (`200` in caso di successo) |
| `scores` | array  | Tutte le 8 categorie, ordinate per probabilità decrescente |
| `key`    | string | Nome della categoria (vedi tabella sotto) |
| `value`  | string | Peso della categoria; somma di tutti i `value` = 1. Formato decimale con virgola (es. `"0,989"`, `"0,0"`) |

### Errori

| HTTP | Messaggio |
|------|-----------|
| 401  | `Formato di autorizzazione non valido.` — token assente o malformato |
| 401  | `Non autorizzato.` — token non valido |

---

## GET /health

Verifica che il servizio sia attivo.

```bash
curl https://ceia.gesan.it/health
```

**Risposta:**

```json
{"status": "ok"}
```

---

## Categorie disponibili (campo `key`)

| # | Categoria |
|---|-----------|
| 1 | Accessibilità/barriere architettoniche/mobilità e trasporti/barriere digitali e media |
| 2 | Altro |
| 3 | Inclusione lavorativa |
| 4 | Istruzione/formazione/inclusione scolastica |
| 5 | Rapporti con datori di lavoro |
| 6 | Salute/sanità/progetto di vita/assistenza domiciliare |
| 7 | Strutture socio-sanitarie |
| 8 | Vita sociale/eventi/sport |

---

## Tempi di risposta

I tempi di risposta tipici sono **~25–40 secondi** per richiesta (il modello valuta tutte e 8 le categorie). Solo la prima richiesta dopo un riavvio del servizio può richiedere più tempo per il caricamento iniziale del modello in memoria.

La macchina virtuale esegue il modello **solo su CPU**. Con una **GPU dedicata**, i tempi scenderebbero a **meno di 1 secondo** per le stesse richieste.

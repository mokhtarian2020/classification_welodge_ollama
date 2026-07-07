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

Classifica un testo valutando **tutte le 5 categorie** con un punteggio pesato. La somma degli `value` in `scores` è **1**.

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
  -d '{"input": "La rampa del municipio è rotta. Utilizzo sedia a rotelle."}'
```

### ESEMPIO OUTPUT

```json
{
  "status": 200,
  "scores": [
    { "key": "Accessibilità, mobilità e tecnologie inclusive (ambito tematico 03)", "value": "0,921" },
    { "key": "Partecipazione sociale, culturale, ricreativa e sportiva (ambito tematico 04)", "value": "0,079" },
    { "key": "Altro", "value": "0,0" },
    { "key": "Istruzione, formazione e inclusione socio-lavorativa (ambito tematico 01)", "value": "0,0" },
    { "key": "Servizi sociosanitari, progetto di vita e assistenza (ambito tematico 02)", "value": "0,0" }
  ]
}
```

| Campo    | Tipo   | Descrizione |
|----------|--------|-------------|
| `status` | int    | Codice di esito (`200` in caso di successo) |
| `scores` | array  | Tutte le 5 categorie, ordinate per probabilità decrescente |
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

| # | Categoria | Contenuto |
|---|-----------|-----------|
| 1 | Istruzione, formazione e inclusione socio-lavorativa (ambito tematico 01) | Istruzione, formazione professionale, inclusione scolastica, rapporti con il sistema educativo e con i datori di lavoro, inclusione lavorativa |
| 2 | Servizi sociosanitari, progetto di vita e assistenza (ambito tematico 02) | Strutture sociosanitarie, progetto di vita individuale, assistenza domiciliare e servizi di supporto alla persona |
| 3 | Accessibilità, mobilità e tecnologie inclusive (ambito tematico 03) | Accessibilità fisica e digitale, barriere architettoniche, mobilità e trasporti, media e servizi digitali |
| 4 | Partecipazione sociale, culturale, ricreativa e sportiva (ambito tematico 04) | Attività sociali, culturali, ricreative e sportive; eventi; associazionismo; turismo accessibile |
| 5 | Altro | Segnalazioni generiche o non classificabili |

---

## Tempi di risposta

I tempi di risposta tipici sono **~2–5 secondi** per richiesta. Solo la prima richiesta dopo un riavvio del servizio può richiedere più tempo per il caricamento iniziale del modello in memoria.

La macchina virtuale esegue il modello **solo su CPU**. Con una **GPU dedicata**, i tempi scenderebbero ulteriormente a **meno di 1 secondo**.

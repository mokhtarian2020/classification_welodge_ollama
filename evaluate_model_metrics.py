import ollama
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Constants
MODEL_NAME = "qwen2.5:3b"

# Load labels
LABELS = pd.read_csv("outputs/label_mapping.csv").squeeze().tolist()

def load_test_data():
    """Load and prepare test data"""
    # Fallback to emails.csv with train-test split
    print("Loading data from emails.csv...")
    df = pd.read_csv('data/emails.csv')
    
    # Use scikit-learn to create a stratified split
    from sklearn.model_selection import train_test_split
    _, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    return test_data

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

NUMBERED_LABELS = {str(i + 1): label for i, label in enumerate(LABELS)}

def format_prompt(text):
    """Format the prompt for classification"""
    numbered = "\n".join(
        f"{i+1}. {label} — {CATEGORY_DESCRIPTIONS[i]}"
        for i, label in enumerate(LABELS)
    )
    return f"""Sei un esperto classificatore di segnalazioni sociali per persone con disabilità.
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
{text}

Numero categoria:"""

def sanitize_prediction(pred: str) -> str:
    """Sanitize the model's prediction to ensure it's a valid label"""
    # Extract first token — model should respond with just a digit
    token = pred.strip().split()[0].rstrip(".").strip() if pred.strip() else ""
    if token in NUMBERED_LABELS:
        return NUMBERED_LABELS[token]
    # Fallback: exact name match
    if token in LABELS:
        return token
    raise ValueError(f"❌ Invalid prediction received: '{pred.strip()}'")

def get_predictions(texts):
    """Get model predictions for the given texts using Ollama"""
    predictions = []
    
    for text in tqdm(texts, desc="Getting predictions"):
        # Format the prompt for this text
        prompt = format_prompt(text)
        
        try:
            # Get prediction from Ollama with system message
            response = ollama.chat(
                model=MODEL_NAME,
                options={"temperature": 0},
                messages=[
                    {
                        "role": "system",
                        "content": f"Sei un classificatore di segnalazioni sociali per persone con disabilità. Rispondi SOLO con un numero intero da 1 a {len(LABELS)}. Nessun testo aggiuntivo."
                    },
                    {"role": "user", "content": prompt.strip()},
                ],
            )
            
            # Sanitize and validate the prediction
            result = response["message"]["content"].strip()
            prediction = sanitize_prediction(result)
            predictions.append(prediction)
            
        except ValueError as e:
            print(f"Warning: {e} Using first label as fallback.")
            predictions.append(LABELS[0])
        except Exception as e:
            print(f"Error during prediction: {e} Using first label as fallback.")
            predictions.append(LABELS[0])
            
    return predictions

def get_label_mapping():
    """Get the mapping between numerical labels and their text representations"""
    return dict(enumerate(LABELS))

def calculate_metrics(y_true, y_pred, label_mapping):
    """Calculate and return various metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, f1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, 
        labels=list(set(y_true) | set(y_pred))  # Use all unique labels from both true and predicted
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Get all unique labels that appear in the data
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Per-class metrics
    class_metrics = {}
    for i, label in enumerate(unique_labels):
        class_metrics[label] = {
            'precision': float(precision[i]),  # Convert numpy types to native Python types
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    metrics['per_class_metrics'] = class_metrics
    metrics['macro_avg'] = {
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1': float(np.mean(f1))
    }
    metrics['confusion_matrix'] = conf_matrix.tolist()
    
    return metrics

def main():
    print("Loading test data...")
    test_data = load_test_data()
    
    print("Getting predictions...")
    predictions = get_predictions(test_data['text'].tolist())
    
    print("\nSample predictions:")
    print("First 5 predictions:", predictions[:5])
    print("First 5 true labels:", test_data['label'].tolist()[:5])
    
    print("\nLoading label mapping...")
    label_mapping = get_label_mapping()
    print("Label mapping:", label_mapping)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(test_data['label'].tolist(), predictions, label_mapping)
    
    # Save metrics to a JSON file
    output_file = 'model_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nMetrics have been saved to {output_file}")
    
    # Print summary metrics
    print("\nSummary Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nMacro Averages:")
    for metric, value in metrics['macro_avg'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPer-class F1 Scores:")
    for label, class_metrics in metrics['per_class_metrics'].items():
        print(f"{label}: {class_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()

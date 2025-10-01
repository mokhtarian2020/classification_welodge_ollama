import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import shutil
import os
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_PATH = "outputs/model"
FEEDBACK_FILE = "feedback.csv"

# Load label mapping from original training
label_mapping = pd.read_csv(os.path.join(MODEL_PATH, "label_mapping.csv"))
label_encoder = LabelEncoder()
label_encoder.classes_ = label_mapping.values.flatten()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with proper configuration
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
).to(device)

# Load and preprocess feedback data
df = pd.read_csv(FEEDBACK_FILE)

# ✅ IMPORTANT: Updated to match Italian column names
df = df.rename(columns={"Testo": "text", "Etichetta_Corretta": "label"})

# Use same label encoding as original training
df["labels"] = label_encoder.transform(df["label"])

# Create dataset with proper column names
dataset = Dataset.from_pandas(df[["text", "labels"]])

# Tokenization function (matches original training)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Training arguments (updated to match original style)
training_args = TrainingArguments(
    output_dir="outputs/model_temp",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    logging_dir="logs_retrain",
    save_strategy="epoch",
    evaluation_strategy="no",  # No evaluation during retraining
    load_best_model_at_end=False
)

# Trainer setup (simplified for retraining)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Retrain the model
trainer.train()

# Save the new model
model.save_pretrained("outputs/model_temp")
tokenizer.save_pretrained("outputs/model_temp")

# Copy label mapping to temp directory
shutil.copy(os.path.join(MODEL_PATH, "label_mapping.csv"), 
            os.path.join("outputs/model_temp", "label_mapping.csv"))

# Replace old model with new one
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
shutil.move("outputs/model_temp", MODEL_PATH)

print("✅ Modello riaddestrato e sostituito con successo!")

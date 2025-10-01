import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"

# Load and prepare data
df = pd.read_csv("data/emails.csv")
label_encoder = LabelEncoder()
df["labels"] = label_encoder.fit_transform(df["label"])  # Changed from label_id to labels

# Create dataset - only include text and labels
dataset = Dataset.from_pandas(df[["text", "labels"]])  # Using labels instead of label_id
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

# Apply tokenization and remove unused columns
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])  # Remove text column after tokenization
dataset = dataset.train_test_split(test_size=0.1)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with explicit problem type
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
).to(device)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="outputs/model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    num_train_epochs=3,
    logging_dir="logs",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# Train and save
trainer.train()
model.save_pretrained("outputs/model")
tokenizer.save_pretrained("outputs/model")
pd.Series(label_encoder.classes_).to_csv("outputs/label_mapping.csv", index=False)
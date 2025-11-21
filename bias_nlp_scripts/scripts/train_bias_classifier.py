import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# === 1. Carica dataset annotato ===
DATA_PATH = "../data/prepared/train_bias_examples_annotated.csv"
MODEL_DIR = "../models/bias_model"
LOG_DIR = "../models/logs"

df = pd.read_csv(DATA_PATH)

# Usa solo le righe con etichetta valida
label_map = {"Low risk": 0, "Some concerns": 1, "High risk": 2}
df = df[df["label"].isin(label_map.keys())].copy()
df["label_id"] = df["label"].map(label_map)

print("Distribuzione etichette nel dataset:")
print(df["label"].value_counts())

# Sottocampiona per non saturare la VM (es. max 3000 righe)
max_samples = 3000
if len(df) > max_samples:
    df = df.sample(n=max_samples, random_state=42)

# === 2. Train/test split ===
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)

train_dataset = Dataset.from_pandas(
    train_df[["text", "label_id"]].rename(columns={"label_id": "label"})
)
test_dataset = Dataset.from_pandas(
    test_df[["text", "label_id"]].rename(columns={"label_id": "label"})
)

# === 3. Tokenizer (DistilRoBERTa, più leggero) ===
tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === 4. Modello ===
model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=3)

# === 5. Metriche ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

# === 6. TrainingArguments ottimizzati ===
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="epoch",
    save_strategy="no",            # evita salvataggi multipli
    num_train_epochs=1,            # una sola epoca per testare stabilità
    per_device_train_batch_size=2, # batch piccolo
    per_device_eval_batch_size=2,
    logging_dir=LOG_DIR,
    logging_steps=50,
    load_best_model_at_end=False,
    report_to=[],
    dataloader_num_workers=0,      # evita thread extra
)

# === 7. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# === 8. Avvia training ===
trainer.train()

# === 9. Salva modello e tokenizer ===
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print(f"Training completato. Modello e tokenizer salvati in {MODEL_DIR}")
print("Etichette: 0=Low risk, 1=Some concerns, 2=High risk")


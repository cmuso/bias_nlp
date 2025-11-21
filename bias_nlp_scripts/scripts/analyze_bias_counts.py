import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === 1. Carica modello e tokenizer ===
MODEL_DIR = "../models/bias_model"   # cartella dove hai salvato il modello
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Mappa etichette
id2label = {0: "Low risk", 1: "Some concerns", 2: "High risk"}

# === 2. Carica dataset ===
DATA_PATH = "../data/prepared/train_bias_examples_annotated.csv"
df = pd.read_csv(DATA_PATH)

# === 3. Funzione di predizione ===
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = outputs.logits.argmax(dim=-1).item()
    return id2label[pred_id]

# === 4. Applica predizione a tutte le righe ===
df["predicted_label"] = df["text"].apply(predict)

# === 5. Distribuzione delle classi ===
print("Distribuzione delle etichette predette:")
print(df["predicted_label"].value_counts())

# === 6. Salva dataset con predizioni ===
df.to_csv("../data/prepared/bias_predictions.csv", index=False)
print("File con predizioni salvato in ../data/prepared/bias_predictions.csv")

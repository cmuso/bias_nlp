import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# 1. Carica dataset
df = pd.read_csv("../data/prepared/outcomes.csv")

# 2. Carica modello RoBERTa
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.eval()

# 3. Etichette
labels = ["Low risk", "Some concerns", "High risk"]

# 4. Funzione batch
def classify_batch(texts):
    inputs = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    preds = torch.argmax(probs, dim=1)
    return [labels[i] for i in preds]

# 5. Applica in batch con barra di avanzamento
batch_size = 32
bias_labels = []

for i in tqdm(range(0, len(df), batch_size), desc="Classificazione"):
    batch_texts = df["text"].iloc[i:i+batch_size].tolist()
    batch_preds = classify_batch(batch_texts)
    bias_labels.extend(batch_preds)

df["bias_label"] = bias_labels

# 6. Salva risultato
df.to_csv("../data/prepared/outcomes_with_bias.csv", index=False)
print("Classificazione completata. File salvato in data/prepared/outcomes_with_bias.csv")

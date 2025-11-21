import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# 1. Carica il modello fine-tunato
model_name = "/home/cmuso/reporting_bias_project/scripts/bias_classifier"
   # cartella dove hai salvato il modello
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. Carica il dataset di outcome
df = pd.read_csv("../data/prepared/outcomes.csv")  # file originale senza bias_label

# 3. Funzione di classificazione
def classify_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return preds

# 4. Classifica in batch
labels_map = {0: "Low risk", 1: "Some concerns", 2: "High risk"}
batch_size = 32
predictions = []

for i in tqdm(range(0, len(df), batch_size), desc="Classificazione"):
    batch_texts = df["text"].iloc[i:i+batch_size].tolist()
    batch_preds = classify_texts(batch_texts)
    predictions.extend(batch_preds)

# 5. Aggiungi colonna con le etichette
df["bias_label"] = [labels_map[p] for p in predictions]

# 6. Salva il file con i risultati
df.to_csv("../data/prepared/outcomes_with_bias.csv", index=False)
print("Classificazione completata. File salvato in data/prepared/outcomes_with_bias.csv")

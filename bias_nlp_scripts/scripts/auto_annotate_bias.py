import pandas as pd
import re

# Carica il dataset
df = pd.read_csv("../data/prepared/train_bias_examples.csv")

# Funzione di assegnazione etichette
def assign_label(text: str) -> str:
    if pd.isna(text):
        return None
    t = text.lower()

    # --- Low risk ---
    if any(kw in t for kw in ["overall survival", "event-free survival", "progression-free survival",
                              "disease-free survival", "recurrence", "objective response", "screening uptake"]):
        return "Low risk"

    # --- Some concerns ---
    if any(kw in t for kw in ["quality of life", "qol", "toxicity", "adverse event", "questionnaire",
                              "survey", "self-reported", "knowledge", "behaviour", "modesty", "fatalism"]):
        return "Some concerns"

    # --- High risk ---
    if any(kw in t for kw in ["feasibility", "acceptability", "adherence", "protocol", "implementation",
                              "peer educator", "exploratory", "biomarker", "sociodemographic", "descriptors"]):
        return "High risk"

    # Default: None (non annotato)
    return None

# Applica la funzione
df["auto_label"] = df["text"].apply(assign_label)

# Se la colonna 'label' è già compilata, mantieni quella; altrimenti usa auto_label
df["label"] = df["label"].fillna(df["auto_label"])

# Salva il nuovo dataset annotato
df.to_csv("../data/prepared/train_bias_examples_annotated.csv", index=False)

print("Annotazione automatica completata. File salvato in ../data/prepared/train_bias_examples_annotated.csv")
print("Distribuzione etichette:")
print(df["label"].value_counts())

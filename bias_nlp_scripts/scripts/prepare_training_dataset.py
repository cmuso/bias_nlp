import pandas as pd

# Leggi gli outcome originali
df = pd.read_csv("../data/prepared/outcomes.csv")

# Normalizza la colonna del testo: scegli il nome giusto se non è 'text'
# Se la tua colonna si chiama 'outcome' o simile, rinominala in 'text'
if "text" not in df.columns:
    # Prova a indovinare la colonna del testo
    for cand in ["outcome", "summary", "sentence", "description"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "text"})
            break

# Aggiungi la colonna 'label' vuota per l'annotazione
df["label"] = ""

# Salva il file pronto per annotare
df.to_csv("../data/prepared/train_bias_examples.csv", index=False)

print("Creato: ../data/prepared/train_bias_examples.csv")
print(df.head())

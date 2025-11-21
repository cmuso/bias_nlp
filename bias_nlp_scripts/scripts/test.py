import pandas as pd

# Carica il dataset annotato
df = pd.read_csv("../data/prepared/train_bias_examples.csv")

# Conta quante righe hai per ciascuna etichetta
print(df["label"].value_counts())

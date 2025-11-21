import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("/home/cmuso/reporting_bias_project/data/prepared/bias_predictions.csv")

# Codifica numerica del rischio
label_map = {"Low risk": 0, "Some concerns": 1, "High risk": 2}
df["risk_score"] = df["predicted_label"].map(label_map)

# Conta outcome per sponsor e filtra quelli con almeno 50
sponsor_counts = df["sponsor"].value_counts()
valid_sponsors = sponsor_counts[sponsor_counts >= 50].index
df_filtered = df[df["sponsor"].isin(valid_sponsors)]

# Calcola distribuzione percentuale per etichetta
sponsor_dist = df_filtered.groupby(["sponsor", "predicted_label"]).size().unstack(fill_value=0)
sponsor_dist_percent = sponsor_dist.div(sponsor_dist.sum(axis=1), axis=0) * 100

# Calcola media e varianza del rischio
risk_stats = df_filtered.groupby("sponsor").agg(
    n_outcome=("predicted_label", "count"),
    pct_high_risk=("predicted_label", lambda x: (x == "High risk").mean() * 100),
    media_rischio=("risk_score", "mean"),
    varianza_rischio=("risk_score", "var")
)

# Unisci percentuali e statistiche
summary = pd.concat([risk_stats, sponsor_dist_percent], axis=1)

# Ordina per percentuale di High risk
summary_sorted = summary.sort_values(by="pct_high_risk", ascending=False)

# Mostra la tabella
print("\nTabella riassuntiva per sponsor (ordinata per % High risk):")
print(summary_sorted.round(2))

# Mostra tabella completa con media e varianza
print(summary_sorted[["n_outcome", "pct_high_risk", "media_rischio", "varianza_rischio"]].round(2).head(10))

# Esporta in LaTeX con i valori reali
print(summary_sorted[["n_outcome", "pct_high_risk", "media_rischio", "varianza_rischio"]].round(2).head(10).to_latex(index=True))




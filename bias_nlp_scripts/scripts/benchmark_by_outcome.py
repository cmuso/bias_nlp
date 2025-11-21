import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("/home/cmuso/reporting_bias_project/data/prepared/bias_predictions.csv")

# Distribuzione per tipo di outcome
outcome_dist = df.groupby(['section', 'predicted_label']).size().unstack(fill_value=0)
outcome_dist_percent = outcome_dist.div(outcome_dist.sum(axis=1), axis=0) * 100

print("\nDistribuzione percentuale per tipo di outcome:")
print(outcome_dist_percent.round(2))

# Grafico stacked bar
outcome_dist_percent.plot(kind='bar', stacked=True, figsize=(8,6), colormap='Set2')
plt.title("Distribuzione del rischio di bias per tipo di outcome")
plt.ylabel("Percentuale")
plt.xlabel("Tipo di outcome")
plt.legend(title="Etichetta")
plt.tight_layout()
plt.savefig("/home/cmuso/reporting_bias_project/data/prepared/outcome_bias_distribution.png")
plt.show()

# bias_nlp

Codice e analisi per la classificazione del rischio di bias negli outcome clinici tramite tecniche di NLP.

## Contenuto del repository

Questo repository contiene esclusivamente gli **script Python** utilizzati per:
- Preprocessare e classificare gli outcome clinici
- Applicare modelli NLP per la valutazione del rischio di bias
- Generare visualizzazioni e tabelle per la reportistica scientifica

## Modelli e dati

Per motivi di spazio e policy GitHub:
- Il file del modello (`model.safetensors`) **non è incluso**
- I dati clinici utilizzati per la classificazione **non sono pubblicati**

Se necessario, il modello può essere rigenerato o scaricato da fonti esterne (es. HuggingFace, Zenodo).  
Contattare l'autore per dettagli o accesso controllato.

## Requisiti

Gli script sono compatibili con Python ≥3.9 e richiedono le seguenti librerie principali:
- `transformers`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tqdm`

Per installare i requisiti:
```bash
pip install -r requirements.txt

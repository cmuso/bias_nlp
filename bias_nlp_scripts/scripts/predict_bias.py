import argparse
import joblib

# Carica il modello addestrato (assumendo che sia stato salvato come bias_model.pkl)
model = joblib.load("../models/bias_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

def predict_text(text):
    X = vectorizer.transform([text])
    y_pred = model.predict(X)
    return y_pred[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Outcome text to classify")
    args = parser.parse_args()

    if args.text:
        label = predict_text(args.text)
        print(f"Predicted label: {label}")

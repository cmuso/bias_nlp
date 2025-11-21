import os
import json
import pandas as pd

def collect_protocols(folder):
    records = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            path = os.path.join(folder, file)
            with open(path) as f:
                protocol = json.load(f)
            nct_id = protocol.get("nct_id")
            title = protocol.get("title")
            sponsor = protocol.get("sponsor")

            # Outcome primari
            for outcome in protocol.get("primary_outcomes", []):
                records.append({
                    "nct_id": nct_id,
                    "section": "primary_outcome",
                    "text": outcome,
                    "title": title,
                    "sponsor": sponsor
                })

            # Outcome secondari
            for outcome in protocol.get("secondary_outcomes", []):
                records.append({
                    "nct_id": nct_id,
                    "section": "secondary_outcome",
                    "text": outcome,
                    "title": title,
                    "sponsor": sponsor
                })
    return records

if __name__ == "__main__":
    folder = "data/protocols"
    records = collect_protocols(folder)
    df = pd.DataFrame(records)
    os.makedirs("data/prepared", exist_ok=True)
    df.to_csv("data/prepared/outcomes.csv", index=False)
    print(f"Salvati {len(df)} outcome in data/prepared/outcomes.csv")

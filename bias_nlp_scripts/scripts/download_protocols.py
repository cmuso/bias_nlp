import requests
import os
import json

def search_nct_ids(query, max_results=20):
    url = "https://clinicaltrials.gov/api/query/study_fields"
    params = {
        "expr": query,
        "fields": "NCTId",
        "min_rnk": 1,
        "max_rnk": max_results,
        "fmt": "json"
    }
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        ids = [field["NCTId"][0] for field in data["StudyFieldsResponse"]["StudyFields"]]
        return ids
    else:
        print("Errore:", response.status_code)
        return []

def fetch_protocol(nct_id):
    url = "https://clinicaltrials.gov/api/query/full_studies"
    params = {
        "expr": nct_id,
        "min_rnk": 1,
        "max_rnk": 1,
        "fmt": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "FullStudiesResponse" in data and data["FullStudiesResponse"]["NStudiesFound"] > 0:
            return data
    return None

def save_protocol(nct_id, data, folder):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{nct_id}.json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Protocollo salvato: {nct_id}")

def main():
    query = "breast cancer"
    nct_ids = search_nct_ids(query, max_results=20)
    print("NCT ID trovati:", nct_ids)

    for nct_id in nct_ids:
        data = fetch_protocol(nct_id)
        if data:
            save_protocol(nct_id, data, "data/protocols")
        else:
            print(f"Nessun protocollo scaricato per {nct_id}")

if __name__ == "__main__":
    main()

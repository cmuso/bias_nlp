import requests
import re
import os
from time import sleep

def search_pubmed_ids(query, max_results=100):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data["esearchresult"]["idlist"]

def fetch_abstract(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "text",
        "rettype": "abstract"
    }
    response = requests.get(url, params=params)
    return response.text

def extract_nct_ids_from_abstracts(pmids):
    nct_ids = set()
    for pmid in pmids:
        abstract = fetch_abstract(pmid)
        matches = re.findall(r'NCT\d{8}', abstract)
        nct_ids.update(matches)
        sleep(0.3)  # Rispetta il rate limit dell'API
    return list(nct_ids)

def save_nct_ids(nct_list, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for nct in nct_list:
            f.write(nct + "\n")
    print(f"Salvati {len(nct_list)} NCT ID in {filepath}")

# ESEMPIO DI USO
if __name__ == "__main__":
    query = "breast cancer clinical trial"
    pmids = search_pubmed_ids(query, max_results=100)
    nct_list = extract_nct_ids_from_abstracts(pmids)
    save_nct_ids(nct_list, "data/nct_ids.txt")


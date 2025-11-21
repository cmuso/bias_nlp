import os
import xml.etree.ElementTree as ET

def extract_nct_ids(folder, keyword="breast cancer"):
    nct_ids = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".xml"):
                path = os.path.join(root, file)
                try:
                    tree = ET.parse(path)
                    root_xml = tree.getroot()
                    nct_id = root_xml.findtext("id_info/nct_id")
                    conditions = [c.text.lower() for c in root_xml.findall("condition") if c.text]
                    if any(keyword in c for c in conditions):
                        nct_ids.append(nct_id)
                except Exception:
                    continue
    return nct_ids

if __name__ == "__main__":
    folder = "data/AllPublicXML"
    ids = extract_nct_ids(folder)
    os.makedirs("data", exist_ok=True)
    with open("data/nct_ids.txt", "w") as f:
        for nct in ids:
            f.write(nct + "\n")
    print(f"Trovati {len(ids)} studi sul breast cancer")

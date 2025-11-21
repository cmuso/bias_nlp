import os
import xml.etree.ElementTree as ET
import json

def load_nct_ids(filepath):
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]

def extract_protocol(nct_id, folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".xml") and nct_id in file:
                path = os.path.join(root, file)
                try:
                    tree = ET.parse(path)
                    root_xml = tree.getroot()
                    protocol = {
                        "nct_id": nct_id,
                        "title": root_xml.findtext("official_title"),
                        "sponsor": root_xml.findtext("sponsors/lead_sponsor/agency"),
                        "primary_outcomes": [o.findtext("measure") for o in root_xml.findall("primary_outcome")],
                        "secondary_outcomes": [o.findtext("measure") for o in root_xml.findall("secondary_outcome")]
                    }
                    return protocol
                except Exception:
                    return None
    return None

def main():
    nct_ids = load_nct_ids("data/nct_ids.txt")
    folder = "data/AllPublicXML"
    os.makedirs("data/protocols", exist_ok=True)

    for nct_id in nct_ids:
        protocol = extract_protocol(nct_id, folder)
        if protocol:
            filepath = os.path.join("data/protocols", f"{nct_id}.json")
            with open(filepath, "w") as f:
                json.dump(protocol, f, indent=2)
            print(f"Protocollo salvato: {nct_id}")
        else:
            print(f"Protocollo non trovato per {nct_id}")

if __name__ == "__main__":
    main()

import os
import json

INPUT_DIR = "crawled_docs_async"
OUTPUT_FILE = "doc_metadata.json"

# INPUT_DIR = "crawled_docs_async_2"
# OUTPUT_FILE = "doc_metadata_2.json"

metadata = []


filenames = sorted(f for f in os.listdir(INPUT_DIR) if f.startswith("doc_") and f.endswith(".json"))

for index, filename in enumerate(filenames):
    path = os.path.join(INPUT_DIR, filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
            metadata.append({
                "index": index,
                "filename": filename,
                "title": doc.get("title", "No title"),
                "url": doc.get("url", "No URL")
            })
            print(f"[OK] [{index}/{len(filenames)}] {filename}: {doc.get('title', 'No title')}")
    except Exception as e:
        print(f"[Błąd] {filename}: {e}")

# Zapis metadanych
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"Zapisano metadane do {OUTPUT_FILE} ({len(metadata)} wpisów)")

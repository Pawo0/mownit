import os
import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

INPUT_DIR = "crawled_docs_async"
OUTPUT_FILE = "vocabulary.json"

# INPUT_DIR = "crawled_docs_async_2"
# OUTPUT_FILE = "vocabulary_2.json"

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
vocab_set = set()

print("Tworzenie słownika...")

articles_count = len(os.listdir(INPUT_DIR))


filenames = sorted(f for f in os.listdir(INPUT_DIR) if f.startswith("doc_") and f.endswith(".json"))


for i, filename in enumerate(filenames):
    path = os.path.join(INPUT_DIR, filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
            text = doc.get("text", "")
            tokens = word_tokenize(text.lower())

            # Filtrowanie
            filtered = [
                word for word in tokens
                if word.isalpha()                # tylko litery
                and word not in stop_words       # bez stopwordów
                and word not in punctuation      # bez znaków interpunkcyjnych
                and len(word) >= 3               # słowa co najmniej 3-literowe
            ]

            vocab_set.update(filtered)
            print(f"[OK] [{i}/{articles_count}] {filename}: {len(filtered)} słów")

    except Exception as e:
        print(f"[Błąd] {filename}: {e}")

# Zapis słownika
sorted_vocab = sorted(vocab_set)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)


print(f"Zapisano slownik z iloscia slow: {len(sorted_vocab)}. Do pliku {OUTPUT_FILE}")

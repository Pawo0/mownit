import os
import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict


from nltk import download
download('punkt')
download('stopwords')

INPUT_DIR = "crawled_docs_async"
OUTPUT_FILE_COUNT = "doc_term_counts.json"
OUTPUT_FILE_FREQ = "doc_freq.json"

# INPUT_DIR = "crawled_docs_async_2"
# OUTPUT_FILE_COUNT = "doc_term_counts_2.json"
# OUTPUT_FILE_FREQ = "doc_freq_2.json"

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

doc_term_counts = []         # lista słowników BoW dla każdego dokumentu
doc_freq = defaultdict(int)  # ile dokumentów zawiera dane słowo

print("Przetwarzanie dokumentów...")

articles_count = len(os.listdir(INPUT_DIR))

filenames = sorted(f for f in os.listdir(INPUT_DIR) if f.startswith("doc_") and f.endswith(".json"))


for i, filename in enumerate(filenames):
    path = os.path.join(INPUT_DIR, filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
            text = doc.get("text", "")
            tokens = word_tokenize(text.lower())

            # Filtrowanie tokenów
            filtered = [
                word for word in tokens
                if word.isalpha()
                and word not in stop_words
                and len(word) >= 3
            ]

            # Zliczanie BoW
            term_count = defaultdict(int)
            for word in filtered:
                term_count[word] += 1

            # Dodajemy do zbioru dokumentów
            doc_term_counts.append(term_count)

            # Zliczanie obecności termów w dokumencie (unikalne słowa)
            for word in set(term_count.keys()):
                doc_freq[word] += 1

            print(f"[OK] [{i}/{articles_count}] {filename}: {len(filtered)} słów, most common: {max(term_count, key=term_count.get)} ({term_count[max(term_count, key=term_count.get)]})")



    except Exception as e:
        print(f"[Błąd] {filename}: {e}")


print(f"Przetworzono {len(doc_term_counts)} dokumentów.")
print(f"Zbudowano {len(doc_freq)} unikalnych termów.")


with open(f"{OUTPUT_FILE_COUNT}", "w", encoding="utf-8") as f:
    json.dump(doc_term_counts, f, ensure_ascii=False, indent=2)

with open(f"{OUTPUT_FILE_FREQ}", "w", encoding="utf-8") as f:
    json.dump(doc_freq, f, ensure_ascii=False, indent=2)

print(f"Zapisano do {OUTPUT_FILE_COUNT} oraz {OUTPUT_FILE_FREQ}")

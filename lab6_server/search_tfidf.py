import json
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize

from nltk import download
download('punkt')
download('stopwords')

# ----- Parametry -----
VOCAB_FILE = "vocabulary.json"
IDF_FILE = "doc_freq.json"
TFIDF_FILE = "tfidf_matrix.npz"
META_FILE = "doc_metadata.json"
K = 10  # liczba wyników

# VOCAB_FILE = "vocabulary_2.json"
# IDF_FILE = "doc_freq_2.json"
# TFIDF_FILE = "tfidf_matrix_2.npz"
# META_FILE = "doc_metadata_2.json"
# K = 10  # liczba wyników

# ----- Wczytanie danych -----
with open(VOCAB_FILE, encoding="utf-8") as f:
    vocabulary = json.load(f)

with open(IDF_FILE, encoding="utf-8") as f:
    doc_freq = json.load(f)

with open(META_FILE, encoding="utf-8") as f:
    metadata = json.load(f)

term_to_index = {term: i for i, term in enumerate(vocabulary)}
idf = {term: np.log(len(metadata) / (doc_freq.get(term, 1))) for term in vocabulary}

A = load_npz(TFIDF_FILE).tocsc()
A = normalize(A, axis=0)

stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> dict:
    tokens = word_tokenize(text.lower())
    tokens = [
        word for word in tokens
        if word.isalpha()
        and word not in stop_words
        and len(word) >= 3
    ]
    counts = {}
    for word in tokens:
        if word in term_to_index:
            counts[word] = counts.get(word, 0) + 1
    return counts

def vectorize_query(counts: dict) -> np.ndarray:
    vec = np.zeros((len(vocabulary),), dtype=np.float32)
    for word, count in counts.items():
        i = term_to_index[word]
        vec[i] = count * idf[word]
    # normalizacja do długości 1 (tak jak A)
    norm = np.linalg.norm(vec)
    if norm != 0:
        vec = vec / norm
    return vec

def search(query: str, k: int = K):
    counts = preprocess(query)
    if not counts:
        print("Zapytanie nie zawiera znanych słów.")
        return
    q = vectorize_query(counts)

    # oblicz podobieństwo kosinusowe q^T * A (macierz sparse)
    scores = q @ A
    top_k_idx = np.argsort(scores)[::-1][:k]

    print(f"\n🔍 Top {k} wyników dla zapytania: \"{query}\"")
    for i, idx in enumerate(top_k_idx, start=1):
        doc = metadata[idx]
        print(f"\n[{i}] {doc['title']}")
        print(f"    URL: {doc['url']}")
        print(f"    Score: {scores[idx]:.4f}")

# ----- Interfejs użytkownika -----
if __name__ == "__main__":
    while True:
        q = input("\nWpisz zapytanie (lub 'exit'): ")
        if q.strip().lower() == "exit":
            break
        search(q)

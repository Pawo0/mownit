import json
import numpy as np
from scipy.sparse import lil_matrix, save_npz
from math import log

INPUT_FILE_COUNT = "doc_term_counts.json"
INPUT_FILE_FREQ = "doc_freq.json"
INPUT_FILE_VOCAB = "vocabulary.json"
OUTPUT_FILE = "tfidf_matrix.npz"

# INPUT_FILE_COUNT = "doc_term_counts_2.json"
# INPUT_FILE_FREQ = "doc_freq_2.json"
# INPUT_FILE_VOCAB = "vocabulary_2.json"
# OUTPUT_FILE = "tfidf_matrix_2.npz"

# Wczytanie danych
with open(f"{INPUT_FILE_COUNT}", encoding="utf-8") as f:
    doc_term_counts = json.load(f)

with open(f"{INPUT_FILE_FREQ}", encoding="utf-8") as f:
    doc_freq = json.load(f)

with open(f"{INPUT_FILE_VOCAB}", encoding="utf-8") as f:
    vocabulary = json.load(f)

N = len(doc_term_counts)          # liczba dokumentów
M = len(vocabulary)               # liczba termów

print(f"Tworzenie macierzy TF-IDF: {M} termów, {N} dokumentów")

# Mapowanie słowo-indeks
term_to_index = {term: i for i, term in enumerate(vocabulary)}

# Oblicz IDF
idf = {}
for term in vocabulary:
    df = doc_freq.get(term, 1)
    idf[term] = log(N / df)

# Tworzymy rzadką macierz TF-IDF
A = lil_matrix((M, N), dtype=np.float32)

for j, doc in enumerate(doc_term_counts):
    for term, count in doc.items():
        i = term_to_index.get(term)
        if i is not None:
            A[i, j] = count * idf[term]
    print(f"[OK] [{j + 1}/{N}] {len(doc)} słów, most common: {max(doc, key=doc.get)} ({doc[max(doc, key=doc.get)]})")

# Zapis do pliku .npz (format scipy sparse)
save_npz(f"{OUTPUT_FILE}", A.tocsc())

print(f"Zapisano macierz TF-IDF do {OUTPUT_FILE}")

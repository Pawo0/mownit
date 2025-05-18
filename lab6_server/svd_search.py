import json
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download

download("punkt")
download("stopwords")

# === Parametry ===
VOCAB_FILE = "vocabulary.json"
IDF_FILE = "doc_freq.json"
TFIDF_FILE = "tfidf_matrix.npz"
META_FILE = "doc_metadata.json"
K_RESULTS = 10
SVD_DIM = 50  # <<---- wymiar k: ile tematów chcesz zachować

# === Wczytywanie ===
with open(VOCAB_FILE, encoding="utf-8") as f:
    vocabulary = json.load(f)

with open(IDF_FILE, encoding="utf-8") as f:
    doc_freq = json.load(f)

with open(META_FILE, encoding="utf-8") as f:
    metadata = json.load(f)

N = len(metadata)
M = len(vocabulary)
term_to_index = {term: i for i, term in enumerate(vocabulary)}
idf = {term: np.log(N / (doc_freq.get(term, 1))) for term in vocabulary}

A = load_npz(TFIDF_FILE).astype(np.float32)

# === SVD ===
print(f"▶ Wykonywanie SVD: macierz {M} × {N} → {SVD_DIM} wymiarów")
U, S, VT = svds(A, k=SVD_DIM)
# Odwróć kolejność (svds sortuje rosnąco)
U = U[:, ::-1]
S = S[::-1]
VT = VT[::-1, :]

# === Macierz w przestrzeni latentnej: Ak = U D V^T ===
#     Dokumenty są w kolumnach VT
Ak_docs = (S[:, None] * VT).T  # n × k
Ak_docs = normalize(Ak_docs, axis=1)

# === Przygotowanie zapytania ===
stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> dict:
    tokens = word_tokenize(text.lower())
    return {
        word: tokens.count(word)
        for word in tokens
        if word.isalpha() and word not in stop_words and word in term_to_index
    }

def vectorize_query(counts: dict) -> np.ndarray:
    q = np.zeros(M, dtype=np.float32)
    for word, count in counts.items():
        i = term_to_index[word]
        q[i] = count * idf[word]
    return q

def project_query(q: np.ndarray) -> np.ndarray:
    # q ∈ R^m → przekształć do przestrzeni latentnej: Uk^T q, następnie pomnóż przez D^-1
    Uk = U
    Sk_inv = np.diag(1 / S)
    qk = Sk_inv @ (Uk.T @ q)
    return qk / np.linalg.norm(qk)

def search_svd(query: str, top_k: int = K_RESULTS):
    counts = preprocess(query)
    if not counts:
        print("❌ Zapytanie nie zawiera znanych słów.")
        return
    q_vec = vectorize_query(counts)
    q_proj = project_query(q_vec)

    # Licz podobieństwo kosinusowe: q_proj ⋅ A_k
    scores = Ak_docs @ q_proj
    top_idx = np.argsort(scores)[::-1][:top_k]

    print(f"\n🔎 Wyniki dla zapytania: \"{query}\"")
    for i, idx in enumerate(top_idx, 1):
        doc = metadata[idx]
        print(f"\n[{i}] {doc['title']}")
        print(f"    URL: {doc['url']}")
        print(f"    Score: {scores[idx]:.4f}")

# === Interfejs użytkownika ===
if __name__ == "__main__":
    while True:
        q = input("\nWpisz zapytanie (lub 'exit'): ")
        if q.strip().lower() == "exit":
            break
        search_svd(q)

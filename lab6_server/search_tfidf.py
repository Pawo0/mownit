import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
import os
import traceback


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_FILE = os.path.join(BASE_DIR, "vocabulary.json")
DOC_FREQ_FILE = os.path.join(BASE_DIR, "doc_freq.json")
TFIDF_FILE_PATH = os.path.join(BASE_DIR, "tfidf_matrix.npz")
META_FILE = os.path.join(BASE_DIR, "doc_metadata.json")
K_DEFAULT = 10


# globalne
vocabulary = []
doc_freq = {}
metadata = []
term_to_index = {}
idf_values = {}
A_matrix_tfidf = None
stop_words_set = set()


def load_tfidf_data():
    global vocabulary, doc_freq, metadata, term_to_index, idf_values, A_matrix_tfidf, stop_words_set
    print("TF-IDF: Ladowanie...")
    try:
        with open(VOCAB_FILE, encoding="utf-8") as f:
            vocabulary = json.load(f)
        if not vocabulary: raise ValueError(f"plik {VOCAB_FILE} jest pusty.")

        with open(DOC_FREQ_FILE, encoding="utf-8") as f:
            doc_freq = json.load(f)
        if not doc_freq: raise ValueError(f"plik {DOC_FREQ_FILE} jest pusty.")

        with open(META_FILE, encoding="utf-8") as f:
            metadata = json.load(f)
        if not metadata: raise ValueError(f"plik {META_FILE} jest pusty.")

        term_to_index = {term: i for i, term in enumerate(vocabulary)}

        N_docs = len(metadata)

        # obliczanie IDF
        # doc_freq.get(term, 1) - użycie 1 jeśli słowo nie ma wpisu, aby uniknąć log(0) lub dzielenia przez 0
        idf_values = {term: np.log(N_docs / (doc_freq.get(term, 1) + 1e-9)) for term in
                      vocabulary}

        A_matrix_tfidf = load_npz(TFIDF_FILE_PATH).tocsc()
        if A_matrix_tfidf.shape[0] != len(vocabulary) or A_matrix_tfidf.shape[1] != N_docs:
            raise ValueError(
                f"Zle wymiary tablicy, oczekiwane: ({len(vocabulary)}, {N_docs}), wczytano: {A_matrix_tfidf.shape}")

        A_matrix_tfidf = normalize(A_matrix_tfidf, norm='l2', axis=0)

        stop_words_set = set(stopwords.words("english"))
        return True

    except FileNotFoundError as e:
        print(f"TF-IDF CRITICAL ERROR: Data file not found: {e}")
        print(traceback.format_exc())
    except json.JSONDecodeError as e:
        print(f"TF-IDF CRITICAL ERROR: JSON decoding error in data file: {e}")
        print(traceback.format_exc())
    except ValueError as e:
        print(f"TF-IDF CRITICAL ERROR: Value error during data loading: {e}")
        print(traceback.format_exc())
    except Exception as e:
        print(f"TF-IDF CRITICAL ERROR: An unexpected error occurred during data loading: {e}")
        print(traceback.format_exc())
    # Jeśli dane się nie załadują, globalne zmienne pozostaną puste/None
    return False


# Wczytaj dane przy imporcie modułu
DATA_LOADED_SUCCESSFULLY_TFIDF = load_tfidf_data()


def preprocess_text_tfidf(text: str) -> dict:
    if not DATA_LOADED_SUCCESSFULLY_TFIDF: return {}
    tokens = word_tokenize(text.lower())
    counts = {}
    for word in tokens:
        if word.isalpha() and word not in stop_words_set and len(word) >= 3 and word in term_to_index:
            counts[word] = counts.get(word, 0) + 1
    return counts


def vectorize_query_tfidf(counts: dict) -> np.ndarray:
    if not DATA_LOADED_SUCCESSFULLY_TFIDF: return np.array([])
    vec = np.zeros((len(vocabulary),), dtype=np.float32)
    if not counts: return vec

    for word, count in counts.items():
        if word in term_to_index:
            i = term_to_index[word]
            vec[i] = count * idf_values.get(word, 0)

    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec


def search(query: str, k: int = K_DEFAULT, return_results: bool = False):
    if not DATA_LOADED_SUCCESSFULLY_TFIDF or A_matrix_tfidf is None:
        print("TF-IDF Error: Dane nie załadowane poprawnie.")
        return [] if return_results else None

    word_counts = preprocess_text_tfidf(query)
    if not word_counts:
        return [] if return_results else None

    query_vector = vectorize_query_tfidf(word_counts)
    if np.linalg.norm(query_vector) < 1e-9:  # Wektor zerowy
        return [] if return_results else None

    scores = query_vector @ A_matrix_tfidf
    if hasattr(scores, "toarray"):
        scores = scores.toarray().flatten()
    elif isinstance(scores, np.matrix):
        scores = np.asarray(scores).flatten()

    num_documents = A_matrix_tfidf.shape[1]
    if len(scores) != num_documents:
        return [] if return_results else None

    actual_k = min(k, len(scores))
    if actual_k <= 0 and len(scores) > 0: actual_k = len(scores)  # Pokaż wszystko, jeśli k jest niepoprawne

    top_k_indices = np.array([], dtype=int)
    if len(scores) > 0:
        # odwracamy kolejność, aby uzyskać sortowanie malejące (najwyższe wyniki pierwsze) i wybieramy k najlepszych
        sorted_indices = np.argsort(scores)[::-1]
        top_k_indices = sorted_indices[:actual_k]

    if return_results:
        results_list = []
        for doc_idx in top_k_indices:
            doc_idx = int(doc_idx)
            # Sprawdzenie, czy indeks dokumentu jest w poprawnym zakresie
            if 0 <= doc_idx < len(metadata):
                doc_meta = metadata[doc_idx]
                score_val = scores[doc_idx]
                # Dodanie wyników do listy wyników
                results_list.append({
                    'title': doc_meta.get('title', f'Document {doc_idx + 1}'),
                    'url': doc_meta.get('url', '#'),
                    'score': float(score_val)
                })
        return results_list
    else:  # Debugowanie w konsoli
        print(f"\n🔍 Top {actual_k} wyniki TF-IDF dla zapytania: \"{query}\"")
        if not top_k_indices.size:
            print("Nie znaleziono wyników.")
        for i, doc_idx in enumerate(top_k_indices, start=1):
            doc_idx = int(doc_idx)
            # Sprawdzenie, czy indeks dokumentu jest w poprawnym zakresie
            if 0 <= doc_idx < len(metadata):
                doc_meta = metadata[doc_idx]
                print(
                    f"[{i}] {doc_meta.get('title', 'N/A')} (Wynik: {scores[doc_idx]:.4f}, URL: {doc_meta.get('url', '#')})")
        return None

# Testowanie w konsoli
if __name__ == "__main__":
    if DATA_LOADED_SUCCESSFULLY_TFIDF:
        print("\nTestowanie modułu TF-IDF z linii komend...")
        test_query = "Donald Trump"
        print(f"Wyszukiwanie dla: '{test_query}'")
        search(test_query, k=3)
        results_api = search(test_query, k=2, return_results=True)
        print(f"\nWyniki API dla '{test_query}': {results_api}")
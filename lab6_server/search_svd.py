import json
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import traceback

# --- Stałe i ścieżki do plików ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_FILE_SVD = os.path.join(BASE_DIR, "vocabulary.json")
DOC_FREQ_FILE_SVD = os.path.join(BASE_DIR, "doc_freq.json")
TFIDF_FILE_FOR_SVD = os.path.join(BASE_DIR, "tfidf_matrix.npz")
META_FILE_SVD = os.path.join(BASE_DIR, "doc_metadata.json")

K_RESULTS_DEFAULT = 10
SVD_DIM_DEFAULT = 50

# --- Globalne zmienne ---
vocabulary_svd = []
metadata_svd = []
term_to_index_svd = {}
idf_values_svd = {}
A_matrix_for_svd = None  # Macierz TF-IDF używana jako podstawa do SVD
stop_words_set_svd = set()
svd_cache = {}  # Cache dla wyników SVD: {k_dim: (U, S, VT, Ak_docs_proj)}
M_vocab_svd = 0
N_docs_svd = 0


def load_svd_base_data():
    global vocabulary_svd, metadata_svd, term_to_index_svd, idf_values_svd, \
        A_matrix_for_svd, stop_words_set_svd, M_vocab_svd, N_docs_svd
    print("SVD: Loading base data...")
    try:
        with open(VOCAB_FILE_SVD, encoding="utf-8") as f:
            vocabulary_svd = json.load(f)
        if not vocabulary_svd: raise ValueError("SVD: Vocabulary is empty.")
        M_vocab_svd = len(vocabulary_svd)

        with open(DOC_FREQ_FILE_SVD, encoding="utf-8") as f:
            doc_freq_svd_local = json.load(f)  # Zmienna lokalna
        if not doc_freq_svd_local: raise ValueError("SVD: Document frequency data is empty.")

        with open(META_FILE_SVD, encoding="utf-8") as f:
            metadata_svd = json.load(f)
        if not metadata_svd: raise ValueError("SVD: Metadata is empty.")
        N_docs_svd = len(metadata_svd)

        term_to_index_svd = {term: i for i, term in enumerate(vocabulary_svd)}
        idf_values_svd = {term: np.log(N_docs_svd / (doc_freq_svd_local.get(term, 1) + 1e-9)) for term in
                          vocabulary_svd}

        A_matrix_for_svd = load_npz(TFIDF_FILE_FOR_SVD).astype(np.float32).tocsc()
        if A_matrix_for_svd.shape[0] != M_vocab_svd or A_matrix_for_svd.shape[1] != N_docs_svd:
            raise ValueError(
                f"SVD: Shape mismatch for base TFIDF matrix. Expected ({M_vocab_svd}, {N_docs_svd}), got {A_matrix_for_svd.shape}")

        stop_words_set_svd = set(stopwords.words("english"))
        print("SVD module base data initialized successfully.")
        return True
    except FileNotFoundError as e:
        print(f"SVD CRITICAL ERROR: Data file not found: {e}")
    except json.JSONDecodeError as e:
        print(f"SVD CRITICAL ERROR: JSON decoding error: {e}")
    except ValueError as e:
        print(f"SVD CRITICAL ERROR: Value error during data loading: {e}")
    except Exception as e:
        print(f"SVD CRITICAL ERROR: An unexpected error occurred: {e}")
    print(traceback.format_exc())
    return False


DATA_LOADED_SUCCESSFULLY_SVD = load_svd_base_data()


def get_or_compute_svd(k_dim_requested: int):
    global A_matrix_for_svd, svd_cache
    if not DATA_LOADED_SUCCESSFULLY_SVD or A_matrix_for_svd is None:
        print("SVD Error: Base data not loaded, cannot compute SVD.")
        return None, None, None, None

    max_possible_k = min(A_matrix_for_svd.shape) - 1
    if max_possible_k <= 0:
        print(f"SVD Error: Matrix dimensions ({A_matrix_for_svd.shape}) too small for SVD.")
        return None, None, None, None

    k_dim = min(k_dim_requested, max_possible_k)
    if k_dim != k_dim_requested:
        print(f"SVD Warning: Requested dim {k_dim_requested} too high. Using {k_dim}.")
    if k_dim <= 0:
        print(f"SVD Error: Effective SVD dimension {k_dim} is invalid.")
        return None, None, None, None

    if k_dim in svd_cache:
        return svd_cache[k_dim]

    print(f"SVD: Computing for dimension: {k_dim} (Matrix: {A_matrix_for_svd.shape})")
    try:
        # Upewnij się, że macierz jest odpowiedniego typu dla svds (float) i formatu (csc/csr)
        U, s_singular_values, VT = svds(A_matrix_for_svd.astype(np.float32), k=k_dim, which='LM')
    except Exception as e:
        print(f"SVD Error during svds computation (k={k_dim}): {e}")
        print(traceback.format_exc())
        return None, None, None, None

    U = U[:, ::-1]
    s_singular_values = s_singular_values[::-1]
    VT = VT[::-1, :]

    # Ak_docs_projection: (N_docs, k_dim) - każdy wiersz to reprezentacja dokumentu
    Ak_docs_projection = (s_singular_values[:, None] * VT).T
    Ak_docs_projection = normalize(Ak_docs_projection, norm='l2', axis=1)

    svd_cache[k_dim] = (U, s_singular_values, VT, Ak_docs_projection)
    return U, s_singular_values, VT, Ak_docs_projection


def preprocess_text_svd(text: str) -> dict:
    if not DATA_LOADED_SUCCESSFULLY_SVD: return {}
    tokens = word_tokenize(text.lower())
    counts = {}
    for word in tokens:
        if word.isalpha() and word not in stop_words_set_svd and len(word) >= 3 and word in term_to_index_svd:
            counts[word] = counts.get(word, 0) + 1
    return counts


def vectorize_query_for_svd_projection(counts: dict) -> np.ndarray:
    if not DATA_LOADED_SUCCESSFULLY_SVD: return np.array([])
    vec = np.zeros((M_vocab_svd,), dtype=np.float32)
    if not counts: return vec
    for word, count in counts.items():
        if word in term_to_index_svd:
            i = term_to_index_svd[word]
            vec[i] = count * idf_values_svd.get(word, 0)
    return vec


def project_query_to_svd_space(query_tfidf_vector: np.ndarray, U_k: np.ndarray, S_k_values: np.ndarray) -> np.ndarray:
    # D_k^-1 * U_k^T * q
    # S_k_values (k_dim,)
    # U_k (M_vocab, k_dim)
    # query_tfidf_vector (M_vocab,)

    # Zabezpieczenie przed dzieleniem przez zero/bardzo małe wartości singularne
    s_k_inv_diag = np.zeros_like(S_k_values, dtype=np.float32)
    valid_s_indices = S_k_values > 1e-9  # Próg
    if not np.any(valid_s_indices):
        print("SVD Warning: All singular values for projection are near zero.")
        return np.array([], dtype=np.float32)  # Pusty wektor, jeśli nie da się podzielić

    s_k_inv_diag[valid_s_indices] = 1.0 / S_k_values[valid_s_indices]

    # U_k.T @ query_tfidf_vector  -> (k_dim, M_vocab) @ (M_vocab,) -> (k_dim,)
    projected_q = U_k.T @ query_tfidf_vector
    q_k = projected_q * s_k_inv_diag  # Mnożenie element-wise

    norm_qk = np.linalg.norm(q_k)
    if norm_qk > 1e-9:
        q_k = q_k / norm_qk
    else:  # Jeśli norma jest bliska zeru, zwróć wektor zerowy o odpowiednim kształcie
        q_k = np.zeros_like(q_k)
    return q_k


def search_svd(query: str, top_k: int = K_RESULTS_DEFAULT, svd_dim_override: int = None, return_results: bool = False):
    if not DATA_LOADED_SUCCESSFULLY_SVD:
        print("SVD Error: Base data not loaded. Cannot perform search.")
        return [] if return_results else None

    current_svd_dim = svd_dim_override if svd_dim_override is not None else SVD_DIM_DEFAULT
    if current_svd_dim <= 0:  # Dodatkowa walidacja
        print(f"SVD Error: Invalid svd_dim_override: {current_svd_dim}")
        return [] if return_results else None

    U_k, s_k_values, _, Ak_docs_proj = get_or_compute_svd(current_svd_dim)
    if U_k is None or s_k_values is None or Ak_docs_proj is None:
        print(f"SVD computation/retrieval failed for dim {current_svd_dim}.")
        return [] if return_results else None

    word_counts = preprocess_text_svd(query)
    if not word_counts: return [] if return_results else None

    query_tfidf_vector = vectorize_query_for_svd_projection(word_counts)
    if np.linalg.norm(query_tfidf_vector) < 1e-9: return [] if return_results else None

    q_k_projected = project_query_to_svd_space(query_tfidf_vector, U_k, s_k_values)
    if q_k_projected.size == 0 or np.linalg.norm(q_k_projected) < 1e-9:  # Pusty lub zerowy wektor po projekcji
        return [] if return_results else None

    # Ak_docs_proj jest (N_docs, k_dim), q_k_projected jest (k_dim,)
    scores_svd = Ak_docs_proj @ q_k_projected

    actual_k_results = min(top_k, len(scores_svd))
    if actual_k_results <= 0 and len(scores_svd) > 0: actual_k_results = len(scores_svd)

    top_k_indices_svd = np.array([], dtype=int)
    if len(scores_svd) > 0:
        sorted_indices_svd = np.argsort(scores_svd)[::-1]
        top_k_indices_svd = sorted_indices_svd[:actual_k_results]

    if return_results:
        results_list_svd = []
        for doc_idx in top_k_indices_svd:
            doc_idx = int(doc_idx)
            if 0 <= doc_idx < len(metadata_svd):
                doc_meta = metadata_svd[doc_idx]
                score_val = scores_svd[doc_idx]
                results_list_svd.append({
                    'title': doc_meta.get('title', f'Document {doc_idx + 1}'),
                    'url': doc_meta.get('url', '#'),
                    'score': float(score_val)
                })
        return results_list_svd
    else:  # Dla testowania z linii komend
        print(f"\n🔍 Top {actual_k_results} SVD (dim={current_svd_dim}) results for: \"{query}\"")
        if not top_k_indices_svd.size: print("No results found.")
        for i, doc_idx in enumerate(top_k_indices_svd, start=1):
            doc_idx = int(doc_idx)
            if 0 <= doc_idx < len(metadata_svd):
                doc_meta = metadata_svd[doc_idx]
                print(
                    f"[{i}] {doc_meta.get('title', 'N/A')} (Score: {scores_svd[doc_idx]:.4f}, URL: {doc_meta.get('url', '#')})")
        return None


if __name__ == "__main__":
    if DATA_LOADED_SUCCESSFULLY_SVD:
        print("\nTesting SVD module from command line...")
        test_svd_query = vocabulary_svd[0] if vocabulary_svd else "example svd"
        print(f"Searching SVD for: '{test_svd_query}' with default dim ({SVD_DIM_DEFAULT})")
        search_svd(test_svd_query, top_k=3)

        print(f"\nSearching SVD for: '{test_svd_query}' with dim=10")
        search_svd(test_svd_query, top_k=2, svd_dim_override=10)

        api_results_svd = search_svd(test_svd_query, top_k=2, svd_dim_override=15, return_results=True)
        print(f"\nAPI SVD results (dim=15) for '{test_svd_query}': {api_results_svd}")
    else:
        print("SVD module could not be tested due to data loading errors.")
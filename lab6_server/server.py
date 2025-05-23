import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from nltk import download
import os
import traceback

# --- Konfiguracja i inicjalizacja ---
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Umożliwia zapytania z innego portu (np. serwera deweloperskiego React)

# Pobranie niezbędnych zasobów NLTK przy starcie aplikacji
try:
    print("Downloading NLTK resources (punkt, stopwords)...")
    download('punkt', quiet=False, raise_on_error=True)
    download('stopwords', quiet=False, raise_on_error=True)
    print("NLTK resources 'punkt' and 'stopwords' are available.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not download NLTK resources: {e}")
    print("The application might not work correctly. Please ensure an internet connection or manually download them.")
    # Rozważ przerwanie, jeśli krytyczne: import sys; sys.exit(1)

# Importowanie logiki wyszukiwania po pobraniu NLTK
try:
    from search_tfidf import search as search_tfidf_func
    from search_svd import search_svd as search_svd_func

    print("Search modules imported successfully.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import search modules: {e}")
    print(
        "Please ensure search_tfidf.py and search_svd.py are in the same directory and all dependencies are installed.")
    search_tfidf_func = None  # type: ignore
    search_svd_func = None  # type: ignore
    # Rozważ przerwanie, jeśli krytyczne: import sys; sys.exit(1)
except Exception as e:  # Inne błędy podczas inicjalizacji modułów (np. brak plików danych)
    print(f"CRITICAL ERROR during search module initialization: {e}")
    print(traceback.format_exc())
    print("The application will likely not function correctly.")
    search_tfidf_func = None  # type: ignore
    search_svd_func = None  # type: ignore
    # Rozważ przerwanie, jeśli krytyczne: import sys; sys.exit(1)


@app.route('/api/search', methods=['POST'])
def handle_search():
    if not search_tfidf_func or not search_svd_func:
        return jsonify(
            {'error': 'Search modules are not available due to an initialization error. Check server logs.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request data is missing or not JSON'}), 400

    query = data.get('query')
    search_type = data.get('type', 'tfidf')
    try:
        k_results = int(data.get('k_results', 10))
        if k_results <= 0:
            return jsonify({'error': 'Number of results (k_results) must be positive'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid value for k_results'}), 400

    svd_dim_param = data.get('svd_dim')

    if not query or not query.strip():
        return jsonify({'error': 'Query is missing or empty'}), 400

    results = []
    try:
        if search_type == 'svd':
            svd_dim_to_use = None
            if svd_dim_param is not None:
                try:
                    svd_dim_to_use = int(svd_dim_param)
                    if svd_dim_to_use <= 0:
                        return jsonify({'error': 'SVD dimension (svd_dim) must be positive'}), 400
                except ValueError:
                    return jsonify({'error': 'Invalid value for svd_dim'}), 400

            results = search_svd_func(
                query,
                top_k=k_results,
                svd_dim_override=svd_dim_to_use,
                return_results=True
            )
        elif search_type == 'tfidf':
            results = search_tfidf_func(
                query,
                k=k_results,
                return_results=True
            )
        else:
            return jsonify({'error': f'Unknown search type: {search_type}'}), 400

        return jsonify(results)

    except Exception as e:
        print(f"Error during search execution: {e}\n{traceback.format_exc()}")
        # Zwróć bardziej generyczny błąd do klienta, szczegóły w logach serwera
        return jsonify({'error': 'An internal server error occurred while processing your request.'}), 500


# Serwowanie aplikacji React (po zbudowaniu)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    # Sprawdź, czy ścieżka nie jest próbą dostępu do API
    if path.startswith('api/'):
        abort(404)  # Nie znaleziono dla ścieżek API, które nie są zdefiniowane

    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):  # type: ignore
        return send_from_directory(app.static_folder, path)  # type: ignore
    else:
        # Jeśli ścieżka nie jest plikiem, serwuj index.html dla routingu po stronie klienta React
        if os.path.exists(os.path.join(app.static_folder, 'index.html')):  # type: ignore
            return send_from_directory(app.static_folder, 'index.html')  # type: ignore
        else:
            return "React app not found. Did you build it and place it in the 'static' folder?", 404


if __name__ == '__main__':
    print("Starting Flask server...")
    # Upewnij się, że katalog static istnieje
    if not os.path.exists('static'):
        print("Warning: 'static' folder not found. Create it to serve the React frontend.")
        # Możesz go utworzyć: os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5454)
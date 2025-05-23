import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:5454'; // Adres URL backendu Flask

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searchType, setSearchType] = useState('tfidf');
  const [svdDim, setSvdDim] = useState(50);
  const [kResults, setKResults] = useState(10);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [lastSearchedQuery, setLastSearchedQuery] = useState('');
  const [lastSearchOptions, setLastSearchOptions] = useState({});

  const handleSearch = useCallback(async (event) => {
    if (event) event.preventDefault();

    if (!query.trim()) {
      setError('Proszę wpisać zapytanie.');
      setResults([]);
      setLastSearchedQuery('');
      return;
    }

    setIsLoading(true);
    setError('');
    // Nie czyść wyników, jeśli chcemy pokazywać stare, dopóki nowe nie przyjdą
    // setResults([]);

    const currentSearchOptions = {
      query,
      type: searchType,
      k_results: parseInt(kResults, 10),
      ...(searchType === 'svd' && { svd_dim: parseInt(svdDim, 10) }),
    };

    setLastSearchOptions(currentSearchOptions); // Zapisz opcje przed wysłaniem

    try {
      const response = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(currentSearchOptions),
      });

      const responseData = await response.json(); // Odczytaj ciało odpowiedzi raz

      if (!response.ok) {
        throw new Error(responseData.error || `Błąd serwera: ${response.status}`);
      }

      setResults(responseData);
      setLastSearchedQuery(query);
    } catch (err) {
      console.error("Błąd podczas wyszukiwania:", err);
      setError(err.message || 'Nie udało się pobrać wyników. Sprawdź konsolę przeglądarki i logi serwera.');
      setResults([]); // Wyczyść wyniki w przypadku błędu
      setLastSearchedQuery(query); // Mimo błędu, zapisz co było szukane
    } finally {
      setIsLoading(false);
    }
  }, [query, searchType, svdDim, kResults]);

  // Efekt do czyszczenia błędu przy zmianie zapytania
  useEffect(() => {
    if (error) setError('');
  }, [query, searchType, svdDim, kResults]);


  return (
    <div className="App">
      <header className="App-header">
        <h1>Wyszukiwarka</h1>
      </header>
      <main>
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Wpisz zapytanie..."
            className="search-input"
            aria-label="Pole zapytania"
          />
          <button type="submit" disabled={isLoading || !query.trim()} className="search-button">
            {isLoading ? 'Szukam...' : 'Szukaj'}
          </button>
        </form>

        <div className="options">
          <div className="option-group">
            <label htmlFor="searchType">Sposób: </label>
            <select
              id="searchType"
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
              aria-label="Wybierz typ wyszukiwania"
              disabled={isLoading}
            >
              <option value="tfidf">TF-IDF</option>
              <option value="svd">TF-IDF + SVD</option>
            </select>
          </div>

          {searchType === 'svd' && (
            <div className="option-group">
              <label htmlFor="svdDim">Wymiar SVD (k): </label>
              <input
                type="number"
                id="svdDim"
                value={svdDim}
                onChange={(e) => setSvdDim(Math.max(1, parseInt(e.target.value, 10) || 1))}
                min="1"
                className="number-input"
                aria-label="Wymiar SVD"
                disabled={isLoading}
              />
            </div>
          )}
          <div className="option-group">
            <label htmlFor="kResults">Liczba wyników: </label>
            <input
              type="number"
              id="kResults"
              value={kResults}
              onChange={(e) => setKResults(Math.max(1, parseInt(e.target.value, 10) || 1))}
              min="1"
              className="number-input"
              aria-label="Liczba wyników"
              disabled={isLoading}
            />
          </div>
        </div>

        {error && <p className="error-message" role="alert">Błąd: {error}</p>}

        {isLoading && <p className="loading-message">Ładowanie wyników dla "{lastSearchOptions.query || query}"...</p>}

        {!isLoading && lastSearchedQuery && (
          <div className="results-section">
            {results.length > 0 ? (
              <>
                <h2>Wyniki dla "{lastSearchedQuery}" ({lastSearchOptions.type}{lastSearchOptions.type === 'svd' ? `, dim: ${lastSearchOptions.svd_dim}` : ''}):</h2>
                <ul className="results-list">
                  {results.map((result, index) => (
                    <li key={`${result.url}-${index}`} className="result-item"> {/* Lepszy unikalny klucz */}
                      <h3>
                        <a href={result.url} target="_blank" rel="noopener noreferrer">
                          {result.title || 'Brak tytułu'}
                        </a>
                      </h3>
                      <p className="result-url">
                        URL: <a href={result.url} target="_blank" rel="noopener noreferrer">{result.url}</a>
                      </p>
                      <p className="result-score">
                        Podobieństwo: {result.score !== undefined ? result.score.toFixed(4) : 'N/A'}
                      </p>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              !error && <p>Brak wyników dla zapytania "{lastSearchedQuery}". Spróbuj innego zapytania lub zmień opcje.</p>
            )}
          </div>
        )}
      </main>
      <footer className="App-footer">
        <p>&copy; Pawel Czajczyk - lab6</p>
      </footer>
    </div>
  );
}

export default App;
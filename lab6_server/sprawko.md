# Laboratorium 6

## Singular Value Decomposition - zastosowania

---

### Plan dzialania

- Na poczatku tworzymy webrawlera, ktory zbiera dane z internetu.
- Z zebranych danych tworzymy slownik wszystkich wystepujacych slow, ktore wykorzystamy do stworzenia macierzy TF-IDF.
- Dla kazdego dokumentu tworzymy bag of words, ktory przetrzymuje informacje o tym, w ilu dokumentach wystepuje dane
  slowo, oraz ile i jakie slowa powtarzaja sie w konkretnym dokumencie.
- Z otrzymanych danych (slownik slow, bag of words i czestosci wystepowania slow) tworzymy macierz TF-IDF.
- Dla ulatwienia dalszych obliczen, z zebranych danych w osobnych plikach zapisujemy do jednego pliku json.
- Tworzymy program ktory pozwala na wprowadznie zapytania, ktore jest przetwarzanie do postaci bag of words. I zwraca
  najbardziej pasujacy artykul.
- Na koniec tworzymy program, ktory dodatkowo wykorzystuje SVD do redukcji wymiarow i bardziej "naturalnego"
  przetwarzania zapytania.

### Dokladny opis

`webcrawler.py` - crawler zaczyna od określonej strony początkowej (w tym przypadku CNN dotyczącej Donalda Trumpa). Przy
pomocy biblioteki `BeautifulSoup` przeszukuje stronę w poszukiwaniu linków do innych artykułów. Następnie, dla każdego z
tych linków, pobiera treść artykułu i zapisuje go w formacie JSON. Okazało się, że crawler jak zaczal w domenie CNN tak
do konca nie udalo mu sie z niej wyjsc.

`create_vocabluary.py` - Program iteruje po wszystkich artykułach i tokenizuje je przy uzyciu funkcji word_tokenize z
biblioteki NLTK. Nastepnie filtrowane sa slowa, ktore nie sa slowami ktore naleza do listy stopwords. I spelniaja kilka
dodatkowych warunkow. Na koniec zapisuje do pliku `vocabluary.json` utworzony slownik unikalnych slow.

`bag_of_words.py` - Rowniez iteruje po wszystkich artykułach i tokenizuje je przy uzyciu funkcji word_tokenize z
biblioteki NLTK. Nastepnie tworzy bag of words, ktory przetrzymuje informacje o tym, w ilu dokumentach wystepuje dane
slowo (plik `doc_frq.json`), oraz ile i jakie slowa wystepuja w konkretnym dokumencie (plik `doc_term_counts.json`).

`build_tfidf.py` - Program na podstawie wczesniej stworzonych plikow `vocabluary.json`, `doc_frq.json` i
`doc_term_counts.json`. Dla kazdego slowa w slowniku oblicza jego IDF w celu nadania mu wagi. Nastepnie tworzy macierz
TF-IDF o wymiarach `(liczba_slow, liczba_dokumentow)`, gdzie kazda wartosc w macierzy to iloczyn TF i IDF dla danego
slowa w danym dokumencie. Na koniec zapisuje ja do pliku `tfidf_matfix.npz`, zebysmy mogli pozniej szybko z niej
korzystac przy porowananiu z zapytaniem

`create_one_file_with_data.py` - Program, ktory tworzy jeden plik z wszystkimi danymi. Takie trzymanie danych pomoże nam
później w ich łatwiejszym przetwarzaniu i utrzymaniu kolejnosci dzieki nadaniu im indeksow.

`search_tfidf.py` - Program wczytuje wczesniej utorzona macierz `tfidf_matfix.npz` i przetwarza wprowadzone zapytanie
przetwarzajac je na wektor, a nastepnie oblicza podobienstwo cosinowe pomiedzy wektorem zapytania a wektorami macierzy.
Na koniec zwracamy k najbardziej podobnych artykulow do zapytania.

`search_tfidf_svd.py` - Dziala dokladnie tak samo jak `search_tfidf.py`, ale dodatkowo wykorzystuje SVD do redukcji
wymiarow macierzy TF-IDF. Dzieki czemu uzyskujemy lepsze wyniki, poniewaz redukcja wymiarow pozwala na lepsze wykrycie
powiazan miedzy slowami. Trzymamy tez w pamiecy wczesniej obliczone SVD, zeby nie obliczac go za kazdym razem przy
korzystaniu z api
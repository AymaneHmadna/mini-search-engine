import os
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stop_words.update(stopwords.words("french"))
stop_words.update({
    "would", "could", "also", "one", "know", "get",
    "like", "thank", "im", "ive", "fig", "figure",
    "table", "al", "pp", "vol", "et", "ibid", "ref",
    "dun", "dune", "comm", "cett", "peut", "tout",
    "tous", "etre", "dont", "cela", "cette", "celui",
    "celle", "ceux", "quand", "bien", "alors", "ainsi",
    "donc", "mais", "avoir", "faire", "tres", "encore",
    "toute", "apres", "avant", "entre", "selon", "lors",
    "depuis", "nbsp", "amp", "quot",
})
stemmer_en = SnowballStemmer("english")
stemmer_fr = SnowballStemmer("french")
FRENCH_WORDS = {"le","la","les","de","du","des","un","une","est","avec","pour"}
def preprocess(text):
    words = word_tokenize(text.lower())
    is_french = any(w in FRENCH_WORDS for w in words)
    stemmer = stemmer_fr if is_french else stemmer_en
    clean_words = [
        w for w in words
        if w.isalpha() and w not in stop_words and 1 < len(w) < 25
    ]
    return [stemmer.stem(w) for w in clean_words]
input_folder = "processed/clean_texts"
documents = []
filenames = []
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".txt"):
        path = os.path.join(input_folder, filename)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            documents.append(text)
            filenames.append(filename)
        except Exception as e:
            print(f"Error {filename}: {e}")
tokenized_docs = [preprocess(doc) for doc in documents]
string_docs = [" ".join(tokens) for tokens in tokenized_docs]
with open("processed/filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)
with open("processed/processed_documents.pkl", "wb") as f:
    pickle.dump(string_docs, f)
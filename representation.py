import numpy as np
import pickle
from gensim.models import Word2Vec
with open("processed/processed_documents.pkl", "rb") as f:
    processed_documents = pickle.load(f)
sentences = [doc.split() for doc in processed_documents]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
w2v_model.save("processed/word2vec.model")
def get_document_vector(doc_tokens, model, vector_size=100):
    valid_words = [word for word in doc_tokens if word in model.wv]
    if not valid_words:
        return np.zeros(vector_size)
    word_vectors = np.array([model.wv[word] for word in valid_words])
    return word_vectors.mean(axis=0)
doc_matrix = np.array([get_document_vector(tokens, w2v_model) for tokens in sentences])
with open("processed/w2v_matrix.pkl", "wb") as f:
    pickle.dump(doc_matrix, f)
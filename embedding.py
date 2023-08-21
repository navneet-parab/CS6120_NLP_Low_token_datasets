from typing import Callable, List
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import os

Embedding = Callable[[List[str]], np.ndarray]

class NeuralVectorizer:

    def __init__(self, embed_size = 100) -> None:

        self.embed_size = embed_size
        path = os.path.join(os.path.dirname(__file__), 
                        f"glove.6B/glove.6B.{embed_size}d.txt")
        
        self.word_embeddings = self._load_glove_dict(path)

    def _load_glove_dict(self, path) -> dict:
        word_embeddings = {}
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.strip().split()
                word = values[0]
                embedding = np.array([float(val) for val in values[1:]])
                word_embeddings[word] = embedding

        print("glove file loaded")
        return word_embeddings
    
    def transform(self, tokens: List[str]):
        embeddings = np.zeros((len(tokens), self.embed_size))
        
        for idx, sentence in enumerate(tokens):
            words = sentence.split()
            num_words = len(words)
            sentence_embedding = np.zeros(self.embed_size)
            for word in words:
                if word in self.word_embeddings:
                    sentence_embedding += self.word_embeddings[word]
            if num_words > 0:
                sentence_embedding /= num_words

            embeddings[idx] = sentence_embedding
        return embeddings.astype("float16")

def bag_of_words(tokens: List[str], fit_corpus = None) -> np.ndarray:
    vectorizer = CountVectorizer()
    if fit_corpus is None:
        bow_matrix = vectorizer.fit_transform(tokens)
        return bow_matrix.toarray().astype("int16"), vectorizer
    else:
        vectorizer = vectorizer.fit(fit_corpus)
        bow_matrix = vectorizer.transform(tokens)
        return bow_matrix.toarray().astype("int16"), vectorizer

def tf_idf(tokens: List[str], fit_corpus = None) -> np.ndarray:
    vectorizer = TfidfVectorizer()
    if fit_corpus is None:
        tfidf_matrix = vectorizer.fit_transform(tokens)
        return tfidf_matrix.toarray().astype("float16"), vectorizer
    else:
        vectorizer = vectorizer.fit(fit_corpus)
        tfidf_matrix = vectorizer.transform(tokens)
        return tfidf_matrix.toarray().astype("float16"), vectorizer



def neural(tokens: List[str], fit_corpus = None, embed_size = 100) -> np.ndarray:
    
    vectorizer = NeuralVectorizer()
    embeddings_matrix = vectorizer.transform(tokens)

    return embeddings_matrix, vectorizer
    
    
    
    

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import json
from embedding import NeuralVectorizer
from utils import preprocess_text


class DataModule:
    def __init__(self, file_paths, embedding_type='count', 
                 vectorizer = None ,is_test = False):
        self.file_paths = file_paths
        self.embedding_type = embedding_type
        self.label_encoder = LabelEncoder() # type: ignore
        self.is_test = is_test
        self.vectorizer = vectorizer
        self._load()
        self._initiate_dataset()

    def _load(self):
        self.corpus, self.labels = [], []
        for path in self.file_paths:
            with open(path, 'r') as file:
                for item in file.readlines():
                    self.corpus.append(json.loads(item)['text'])
                    self.labels.append(json.loads(item)['classification'])


        self.corpus = [preprocess_text(t) for t in self.corpus]
        self.label_encoder.fit(self.labels)
        self.labels = self.label_encoder.transform(self.labels)

    def _initiate_dataset(self):

        self._train_val_split()

        
        if self.vectorizer is None:
            if self.embedding_type == 'count':
                self.vectorizer = CountVectorizer()
            elif self.embedding_type == 'tfidf':
                self.vectorizer = TfidfVectorizer()
            elif self.embedding_type == 'neural':
                self.vectorizer = NeuralVectorizer()
            else:
                raise ValueError("Invalid embedding_type. Use 'count' or 'tfidf'.")
            
        if not self.is_test:
            self.train_dataset = TextClassificationDataset(self.train_corpus, 
                                                    self.train_labels, 
                                                    self.vectorizer)
            
            self.test_dataset = TextClassificationDataset(self.test_corpus, 
                                                  self.test_labels, 
                                                  self.vectorizer)
        else:
            self.test_dataset = TextClassificationDataset(self.test_corpus, 
                                                  self.test_labels, 
                                                  self.vectorizer, is_trained = True)

        

    def _train_val_split(self, train_size=0.8):

        if not self.is_test:
        
            self.train_corpus, self.test_corpus, self.train_labels, self.test_labels = train_test_split(self.corpus, self.labels, 
                                                        train_size=train_size,
                                                            random_state=65)
        else:
            self.test_corpus, self.test_labels = self.corpus, self.labels

    def create_data_loaders(self, batch_size=32, num_workers=0):
        if not self.is_test:
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, 
                                    num_workers=num_workers, shuffle=True)
            val_loader = DataLoader(self.test_dataset, batch_size=batch_size,
                                    num_workers=num_workers)
            return train_loader, val_loader
        else:
            val_loader = DataLoader(self.test_dataset, batch_size=batch_size,
                                    num_workers=num_workers)
            return None, val_loader

    def get_label_encoder(self):
        return self.label_encoder
    
    def get_vocab_size(self):
        try:
            return self.train_dataset.get_vocab_size()
        except:
            print("please initialize dataset first")

    def get_vectorizer(self):
        return self.vectorizer
class TextClassificationDataset(Dataset):
    def __init__(self, corpus, labels, embedder, is_trained = False):
        self.corpus = corpus
        self.labels = labels
        self.vectorizer = embedder
        self.is_trained = is_trained
        # Initialize the embedding class (CountVectorizer or TfidfVectorizer)

        # Fit the vectorizer on the corpus
        if not self.is_trained:
            try:
                self.vectorizer.fit(self.corpus)
            except:
                print("Using neural embedding")

    def get_vocab_size(self):
        try:
            return len(self.vectorizer.vocabulary_)
        except:
            return self.vectorizer.embed_size
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        label = self.labels[idx]
        try:
        # Transform the text using the fitted vectorizer
            transformed_text = self.vectorizer.transform([text]).toarray()
        except:
            transformed_text = self.vectorizer.transform([text])

        # Convert to PyTorch tensors
        transformed_text = torch.tensor(transformed_text, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return transformed_text, label

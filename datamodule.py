import numpy as np
import os
import json
import sklearn
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from embedding import tf_idf, bag_of_words, neural
from utils import preprocess_text
from sklearn.preprocessing import LabelEncoder
import json

class BaseDataModule(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_data(self, as_array = True):
        pass

    

class DataModule(BaseDataModule):

    def __init__(self, paths:list[str], embedding = bag_of_words, fit_corpus = None) -> None:
        """
        Paths: A list of json paths designed to load in multiple paths(or folds)
        Embedding: The embedding function to embed the training
        fit_corpus: If the embedding needs a separate embedding dataset
        """
        super().__init__()
        self.paths = paths
        self.texts = None
        self.labels = None
        self.fit_corpus = None
        self.embedding = embedding
        self.label_enc = LabelEncoder()
        

    def _embed(self, texts):
        matrix, self.vectorizer = self.embedding(texts, 
                                                fit_corpus = self.fit_corpus)
        
        return matrix



    def load(self) -> None:

        try:
            dataset = []
            for path in self.paths:
                with open(path, 'r') as file:
                    for line in file.readlines():
                        dataset.append(json.loads(line))
        except:
            raise FileNotFoundError
        
        self.texts = [preprocess_text(d['text']) for d in dataset]
        self.labels = [d['classification'] for d in dataset]


        self.x = self._embed(self.texts)
        self.y = self.label_enc.fit_transform(np.array(self.labels))

        print("data loaded and embedded") 
        
    def get_data(self, split=True):
        if split:
            x_train, x_test, y_train, y_test = train_test_split(self.x, 
                                                                self.y, 
                                                                test_size=0.1, 
                                                                random_state=44)
        
            return x_train, x_test, y_train, y_test
        else:
            return self.x, self.y
        
    def get_classes_from_data(self, y):
        return self.label_enc.inverse_transform(y)
    
    def get_name(self):
        if self.paths[0].lower().__contains__("amazon"):
            return "amazon"
        else:
            return "news"
        
    def generate_holdout(self, path):

        #Load the holdout data
        #Reading holdout set
        try:
            test_dataset = []
            
            with open(path, 'r') as file:
                for line in file.readlines():
                    test_dataset.append(json.loads(line))
        except:
            raise FileNotFoundError

        self.test_texts = [preprocess_text(d['text']) for d in test_dataset]
        self.test_labels = [d['classification'] for d in test_dataset]
        try:
            self.x_test = self.vectorizer.transform(self.test_texts).toarray()
        except:
            self.x_test = self.vectorizer.transform(self.test_texts)
        self.y_test = self.label_enc.transform(np.array(self.test_labels))

        print("holdout generated")

        return self.x_test, self.y_test


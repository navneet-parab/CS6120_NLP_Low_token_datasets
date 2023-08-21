from abc import ABC, abstractmethod
from typing import List

from nltk import PorterStemmer, word_tokenize


class Tokenizer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass


class BasicTokenizer(Tokenizer):
    def __init__(self):
        self.stemmer = PorterStemmer()

    @property
    def name(self) -> str:
        return "basic-tokenizer"

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for token in word_tokenize(text):
            if token.isalpha():
                tokens.append(self.stemmer.stem(token, to_lowercase=True))

        return tokens

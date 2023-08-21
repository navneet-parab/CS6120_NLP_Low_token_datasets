import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, TypeVar, Tuple, Union

from dataclasses_json import dataclass_json
from tqdm import tqdm

from tokenizer import Tokenizer
from sklearn.model_selection import train_test_split


@dataclass
class Data(ABC):
    documents: list


@dataclass_json
@dataclass
class OrdinalData(Data):
    @dataclass
    class Document:
        text: str
        tokens: List[str]
        classification: int
        metadata: dict = field(default_factory=dict)

        def to_dict(self) -> dict:
            return {
                "text": self.text,
                "tokens": self.tokens,
                "classification": self.classification,
                "metadata": self.metadata
            }

        @staticmethod
        def from_dict(d):
            return ClassificationData.Document(
                text=d["text"],
                tokens=d["tokens"],
                classification=d["classification"],
                metadata=d["metadata"],
            )

    documents: List[Document]

    def to_dict(self) -> dict:
        return {
            "documents": [doc.to_dict() for doc in self.documents]
        }

    @staticmethod
    def from_dict(d):
        return ClassificationData(documents=[ClassificationData.Document.from_dict(doc) for doc in d["documents"]])


@dataclass
class ClassificationData(Data):
    @dataclass
    class Document:
        text: str
        tokens: List[str]
        classification: str
        metadata: dict = field(default_factory=dict)

        def to_dict(self) -> dict:
            return {
                "text": self.text,
                "tokens": self.tokens,
                "classification": self.classification,
                "metadata": self.metadata
            }

        @staticmethod
        def from_dict(d):
            return ClassificationData.Document(
                text=d["text"],
                tokens=d["tokens"],
                classification=d["classification"],
                metadata=d["metadata"],
            )

    documents: List[Document]

    def to_dict(self) -> dict:
        return {
            "documents": [doc.to_dict() for doc in self.documents]
        }

    @staticmethod
    def from_dict(d):
        return ClassificationData(documents=[ClassificationData.Document.from_dict(doc) for doc in d["documents"]])


@dataclass
class DataSet(ABC):
    training_data: Data
    validation_data: Data
    test_data: Data


@dataclass_json
@dataclass
class OrdinalDataSet(DataSet):
    training_data: OrdinalData
    validation_data: OrdinalData
    test_data: OrdinalData

    def to_dict(self) -> dict:
        return {
            "training_data": self.training_data.to_dict(),
            "validation_data": self.validation_data.to_dict(),
            "test_data": self.test_data.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        return ClassificationDataSet(
            training_data=ClassificationData.from_dict(d["training_data"]),
            validation_data=ClassificationData.from_dict(d["validation_data"]),
            test_data=ClassificationData.from_dict(d["test_data"]),
        )


@dataclass_json
@dataclass
class ClassificationDataSet(DataSet):
    training_data: ClassificationData
    validation_data: ClassificationData
    test_data: ClassificationData

    def to_dict(self) -> dict:
        return {
            "training_data": self.training_data.to_dict(),
            "validation_data": self.validation_data.to_dict(),
            "test_data": self.test_data.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        return ClassificationDataSet(
            training_data=ClassificationData.from_dict(d["training_data"]),
            validation_data=ClassificationData.from_dict(d["validation_data"]),
            test_data=ClassificationData.from_dict(d["test_data"]),
        )


DATA_DIR = Path(__file__).parent / "data"


T = TypeVar("T")


def __split_data(data: List[T]) -> Tuple[List[T], List[T], List[T]]:
    train, test_validation = train_test_split(data, test_size=0.2, random_state=0)
    test, validation = train_test_split(test_validation, test_size=0.5, random_state=0)
    return train, test, validation


def load_amazon(tokenizer: Tokenizer) -> OrdinalDataSet:
    """
    Load the Amazon review dataset
    """
    preprocessed_path = DATA_DIR / f"amazon-digital-music-{tokenizer.name}.json"
    if preprocessed_path.exists():
        with open(preprocessed_path) as file:
            data_dict = json.load(file)
        return OrdinalDataSet.from_dict(data_dict)
    else:
        print(f"No preprocessed data found at {preprocessed_path}")
        reviews = []
        with open(DATA_DIR / "amazon-digital-music-raw.json") as file:
            for line in tqdm(file.readlines(), desc="Tokenizing reviews"):
                review = json.loads(line)

                score = int(review["overall"])
                if "reviewText" in review:  # apparently some reviews don't contain reviewText?
                    text = review["reviewText"]
                    tokens = tokenizer.tokenize(text)

                    reviews.append(
                        OrdinalData.Document(text=text, tokens=tokens, 
                                             classification=score)
                    )

        train, test, validation = __split_data(reviews)
        data_set = OrdinalDataSet(
            training_data=OrdinalData(train),
            validation_data=OrdinalData(validation),
            test_data=OrdinalData(test),
        )
        preprocessed_path.parent.mkdir(exist_ok=True, parents=True)
        with open(preprocessed_path, "w") as file:
            json.dump(data_set.to_dict(), file)
        return data_set


def load_news(tokenizer: Tokenizer) -> ClassificationDataSet:
    """
    Load the news classification dataset
    """
    preprocessed_path = DATA_DIR / f"news-categories-{tokenizer.name}.json"
    if preprocessed_path.exists():
        with open(preprocessed_path) as file:
            data_dict = json.load(file)
        return ClassificationDataSet.from_dict(data_dict)
    else:
        print(f"No preprocessed data found at {preprocessed_path}")
        articles = []
        with open(DATA_DIR / "news-categories-raw.json") as file:
            for line in tqdm(file.readlines(), desc="Tokenizing articles"):
                article = json.loads(line)

                headline = article["headline"]
                description = article["short_description"]
                tokens = tokenizer.tokenize(description)
                classification = article["category"]

                articles.append(
                    ClassificationData.Document(text=description, tokens=tokens, classification=classification)
                )

        train, test, validation = __split_data(articles)
        data_set = ClassificationDataSet(
            training_data=ClassificationData(train),
            validation_data=ClassificationData(validation),
            test_data=ClassificationData(test),
        )
        preprocessed_path.parent.mkdir(exist_ok=True, parents=True)
        with open(preprocessed_path, "w") as file:
            json.dump(data_set.to_dict(), file)
        return data_set

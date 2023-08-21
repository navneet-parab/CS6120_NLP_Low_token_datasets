from abc import abstractmethod, ABC
from pathlib import Path
from typing import TypeVar, Type, Optional

from data import DataSet
from embedding import Embedding
from model_kind import ModelKind

T = TypeVar("T", bound="Task")
C = TypeVar("C", bound="Task.Config")


class Task(ABC):
    """
    An interface for interacting with different types of models.
    This can rely on other helper classes, such as a "Model" class
    """

    class Config(ABC):
        """
        A class that stores model hyper-parameters
        """

        @abstractmethod
        def to_dict(self) -> dict:
            """
            Convert the config to a dictionary so that it can be converted to json
            """
            pass

        @classmethod
        @abstractmethod
        def from_dict(cls: Type[C], path:str) -> C:
            """
            Create a config from a dictionary so that it can be loaded from json
            """
            pass

    @classmethod
    @abstractmethod
    def create(
        cls: Type[T], config: Config, kind: ModelKind, embedding: Embedding
    ) -> T:
        """
        Initialize the task (should do the same thing as __init__)
        :param config: the model's configuration
        :param kind: whether the model with produce ordinals or classifications
        :param embedding: which embedding method to use
        """
        pass

    @classmethod
    @abstractmethod
    def load_from_file(cls: Type[T], path: Path) -> T:
        pass

    @abstractmethod
    def save_to_file(self, path: Path) -> None:
        pass

    @abstractmethod
    def prep(self, data: DataSet):
        pass

    @abstractmethod
    def fit(self) -> None:
        """
        Note that parameters that guide training should go in the config object
        """
        pass

class TaskV2(ABC):
    """
        An interface for interacting with different types of models.
        This can rely on other helper classes, such as a "Model" class
    """
    
    @classmethod
    @abstractmethod
    def create(
        cls: Type[T], config: dict, kind: ModelKind, embedding: Embedding
    ) -> T:
        """
        Initialize the task (should do the same thing as __init__)
        :param config: the model's configuration
        :param kind: whether the model with produce ordinals or classifications
        :param embedding: which embedding method to use
        """
        pass

    @classmethod
    @abstractmethod
    def load_from_file(cls: Type[T], path: Path) -> T:
        pass

    @abstractmethod
    def save_to_file(self, path: Path) -> None:
        pass

    @abstractmethod
    def prep(self, data: DataSet):
        pass

    @abstractmethod
    def fit(self) -> None:
        """
        Note that parameters that guide training should go in the config object
        """
        pass
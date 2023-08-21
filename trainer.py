from pathlib import Path
from typing import TypeVar, Type

import embedding
from data import load_amazon
from model_kind import ModelKind
from task import Task

T = TypeVar("T", bound=Task)


def train_model(model_type: Type[Task], config: Task.Config) -> None:
    data = load_amazon()
    task = model_type.create(config, ModelKind.ORDINAL, embedding.bag_of_words)
    task.prep(data)
    task.fit(
        epochs=100,
        eval_frequency=10,
        checkpoint_frequency=0,
        checkpoint_dir=Path("checkpoints"),
    )
    task.save_to_file(Path("model.pth"))

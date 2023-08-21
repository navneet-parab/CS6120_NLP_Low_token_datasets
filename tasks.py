import sys
import os
from dataclasses import dataclass, field
sys.path.append(os.path.dirname(__file__))
from pathlib import Path
from typing import TypeVar, Type, Optional, Union
import embedding
from embedding import Embedding
from data import load_amazon
from data import DataSet
from model_kind import ModelKind
import json
import sklearn
from sklearn.linear_model import LogisticRegression
from datamodule import DataModule
import utils
from utils import classification_scorer
from utils import NpEncoder
import time
# from task.Task import Config

# @dataclass
class Task():
    

    def __init__(self,classifier, datamodule:DataModule, hyperparameters = None) -> None:
        
        
        self.datamodule = datamodule
        self.hyperparameters = hyperparameters
        self.classifier = classifier

    @classmethod
    def load_from_file(path):
        pass

    
    def save_to_file(self, path) -> None:
        pass

    
    def prep(self):
        
        self.x_train, self.x_test, self.y_train, self.y_test = self.datamodule.get_data() # type: ignore

    
    def fit(self) -> None:
        """
        Note that parameters that guide training should go in the config object
        """
        print("training model")
        start = time.time()
        self.classifier.fit(self.x_train, self.y_train)
        self.fit_time = time.time() - start
        print("model trained")

    def predict(self, x):

        try:
            pred = self.classifier.predict(x)
        except:
            pred = self.classifier.predict(x.toarray())
        return pred
        
    def score(self, x, y, dump_file = True, path:Union[str,None] = None):
        
        y_pred = self.predict(x)

        cls_report, confusion_matrix = classification_scorer(y, y_pred)
        if not dump_file:
            return cls_report, confusion_matrix
        else:
            performance_dict = {}
            performance_dict['cls_report'] = cls_report
            performance_dict['confusion_matrix'] = confusion_matrix
            performance_dict['classes'] = self.datamodule.label_enc.classes_
            performance_dict['time_taken'] = self.fit_time

            text = json.dumps(performance_dict, cls=NpEncoder)
            if path:
                with open(path, 'w') as file:
                    file.write(text)
            else:
                raise ValueError("Did not specify a path while dumping")

    
    
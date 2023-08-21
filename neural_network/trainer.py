import time
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Union
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_data_module import DataModule
from model import SimpleNeuralNetwork
import numpy as np
from utils import classification_scorer, NpEncoder
import json


class Trainer:
    def __init__(self, model, datamodule:DataModule, run_str,
                 hyperparameters:dict = {}) -> None:
        self.model = model
        self.datamodule = datamodule
        self.run_str = run_str
        self.train_dataloader, self.val_dataloader \
            = self.datamodule.create_data_loaders(batch_size=128)
        
        if 'epochs' in hyperparameters:
            self.epochs = hyperparameters['epochs']
        else:
             self.epochs = 3
        if 'lr' in hyperparameters:
            self.lr = hyperparameters['lr']
        else:
             self.lr = 1e-2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(run_str)
        self.validation_interval = 2

        pass
    def fit(self):
        # Training loop
        training_start = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            start = time.time()
            for inputs, labels in self.train_dataloader: #type: ignore
                inputs, labels = inputs.squeeze().to(self.device), labels.to(self.device)  # Move data to the selected device
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / len(self.train_dataloader)#type: ignore
            self.writer.add_scalar('Train Loss', average_loss, epoch)
            if (epoch+1) % self.validation_interval == 0:

                self.model.eval()
                all_labels = np.array([])
                all_preds = np.array([])
                val_loss = 0
                with torch.no_grad():
                    for inputs, labels in self.val_dataloader:
                        inputs, labels = inputs.squeeze().to(self.device), labels.to(self.device)  # Move data to the selected device
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_labels = np.hstack((all_labels, labels.cpu().numpy()))
                        all_preds = np.hstack((all_preds, preds.cpu().numpy()))
                        val_loss += self.loss_fn(outputs, labels).item()
                val_loss /= len(self.val_dataloader)
                accuracy = accuracy_score(all_labels, all_preds)
                self.writer.add_scalar('Validation Accuracy', accuracy, epoch)
                self.writer.add_scalar('Validation Loss', val_loss, epoch)

                print(f'Epoch [{epoch+1}/{self.epochs}] - Train Loss: {average_loss:.4f} - Validation Accuracy: {accuracy:.4f}')
            else:
                time_taken = time.time() - start
                print(f'Epoch [{epoch+1}/{self.epochs}] - Train Loss: {average_loss:.4f} - Time taken: {time_taken}')


        self.fit_time = time.time() - training_start

    def predict(self, test_dataloader):
        self.model.eval()
        all_labels = np.array([])
        all_preds = np.array([])
        
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.squeeze().to(self.device), labels.to(self.device)  # Move data to the selected device
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels = np.hstack((all_labels, labels.cpu().numpy()))
                all_preds = np.hstack((all_preds, preds.cpu().numpy()))


        return all_labels, all_preds
                
        
        

    def score(self, test_datamodule, path=None):
        _, test_dataloader = test_datamodule.create_data_loaders()
        y_true, y_pred = self.predict(test_dataloader)
        cls_report, confusion_matrix = classification_scorer(y_true, y_pred)
        if not path:
            return cls_report, confusion_matrix
        else:
            performance_dict = {}
            performance_dict['cls_report'] = cls_report
            performance_dict['confusion_matrix'] = confusion_matrix
            label_enc = self.datamodule.get_label_encoder()
            performance_dict['classes'] = label_enc.classes_
            performance_dict['time_taken'] = self.fit_time

            text = json.dumps(performance_dict, cls=NpEncoder)
            if path:
                with open(path, 'w') as file:
                    file.write(text)
            else:
                raise ValueError("Did not specify a path while dumping")
        pass


if __name__ == "__main__":


    paths = {"amazon":['../data/amazon_resample_train.json'], 
             "news":["../data/news-resample-0.json", 
                     "../data/news-resample-1.json", 
                     "../data/news-resample-2.json",
                     "../data/news-resample-3.json",
                     "../data/news-resample-4.json",
                     "../data/news-resample-5.json",
                     "../data/news-resample-6.json",
                    "../data/news-resample-7.json",
                     "../data/news-resample-8.json",
                     "../data/news-resample-9.json"]}
    holdout = {"amazon":['../data/amazon_test.json'], 
               "news":['../data/news_test.json']}
    
    embeds = ['count', 'tfidf', 'neural']

    for dataset in ['amazon']:
        for emb in embeds:

            dm = DataModule(paths[dataset], embedding_type=emb)
            train_vectorizer = dm.get_vectorizer()
            test_dm = DataModule(holdout[dataset], embedding_type=emb, 
                                vectorizer=train_vectorizer, is_test=True)

            model = SimpleNeuralNetwork(dm.get_vocab_size(), output_size=5)
            run_str = f"nn-{emb}-{dataset}"
            if emb == 'neural':
                hyper = {"epochs": 15}
            else:
                hyper = {"epochs": 10}
            trainer = Trainer(model, dm, run_str, hyperparameters=hyper)
            trainer.fit()
            trainer.score(test_dm, f"{run_str}.json")

            del model
            del trainer
            del dm
            del test_dm
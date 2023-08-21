import json
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from xgboost import XGBClassifier
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tasks import Task
from datamodule import DataModule
from embedding import bag_of_words, tf_idf, neural


def main():

    # embeddings = ['bow', 'tf-idf', 'neural']
    embed_dict = {
        "bow":bag_of_words,
        "tf-idf":tf_idf,
        # "neural":neural
    }

    cls_dict = {
        # "lr": LogisticRegression,
        # "nb": MultinomialNB,
        "xgb": XGBClassifier

    }
    #This is a list of lists
    #The first list contains datapath for the amazon dataset and the second path contains paths for the news dataset
    # data_paths = [#["data/amazon-digital-music-raw-fold-1.json", "data/amazon-digital-music-raw-fold-2.json"], 
    #               ["data/news-categories-raw-fold-1.json", "data/news-categories-raw-fold-2.json"]]
    # data_paths = [#["data/amazon-digital-music-raw-fold-1.json", "data/amazon-digital-music-raw-fold-2.json"], 
                #   ["data/news-categories-raw-fold-0.json"]]
    # data_paths = [["data/amazon_resample_train.json"]]

    data_paths = [["data/news-resample-3.json", "data/news-resample-4.json"]]
    
    holdout_paths = [#"data/amazon_test.json"
                  "data/news_test.json"]
    for idx, paths in enumerate(data_paths):
        print(idx, paths)
        for clf in cls_dict.keys():
            for embed in embed_dict.keys():
                
                dm = DataModule(paths=paths, embedding=embed_dict[embed])
                dm.load()

                results_path = f"{clf}-{embed}-{dm.get_name()}.json"
                if embed == "neural" and clf == 'nb':
                    clf_obj = GaussianNB()
                else:
                    clf_obj = cls_dict[clf]()
                cls_task = Task(classifier=clf_obj, datamodule=dm)
                print("preparing model")
                cls_task.prep()

                print("training model")
                cls_task.fit()

                x_test, y_test = dm.generate_holdout(holdout_paths[idx])

                cls_task.score(x_test, y_test, dump_file=True, path=results_path)
                print(f"Done with file : {results_path}")

                #Delete to free some memory
                del dm
                del cls_task
                del x_test
                del y_test
                del clf_obj
    pass

if __name__ == '__main__':

    main()
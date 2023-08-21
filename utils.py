import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import (precision_score, 
                             recall_score, 
                             classification_report, 
                             f1_score, 
                             confusion_matrix)
import numpy as np
import json

STOPWORDS = stopwords.words("english")
## Adding some stopwords relevant to wikipedia

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def preprocess_text(text):
        '''
        Input:
            text: a string containing a text.
        Output:
            text_cleaned: a processed text. 

        '''
        text = re.sub(r'[^\w\s]', ' ', text)
        text  = re.sub("\s\s+", " ", text)
        #regex to remove links
        url_pat = "https?:\/\/.*?[\s+]"
        text = re.sub(url_pat, " ", text)

        #regex to remove email
        email_pat = r'[\w\.-]+@[\w\.-]+'
        text = re.sub(email_pat, " ", text)

        
        text  = re.sub("\s\s+", " ", text)

        text = text.lower()
        tokens = word_tokenize(text)
        
        #remove stopwords
        tokens = [w for w in tokens if w not in STOPWORDS]

        text_cleaned = " ".join(tokens)
    
        
    
        return text_cleaned

def classification_scorer(y_true:np.ndarray, y_pred:np.ndarray, labels = None):
    
    cls_report = classification_report(y_true = y_true,
                                        y_pred=y_pred,
                                        labels=labels, output_dict=True)

    c_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    return cls_report, c_mat
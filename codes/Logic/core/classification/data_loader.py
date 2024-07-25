import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



import sys
import os
import string
# Add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Add the directory containing indexes_enum.py to sys.path
fasttext_model_dir = os.path.join(project_root, 'Logic', 'core','word_embedding','fasttext_model' )
sys.path.append(fasttext_model_dir)








from Logic.core.word_embedding.fasttext_model import preprocess_text, FastText



class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        df = pd.read_csv(self.file_path)
        self.review_tokens = [preprocess_text(text).split() for text in df['review']]
        self.sentiments = df['sentiment'].values


        self.fasttext_model = FastText()
        self.fasttext_model.load_model(path='C:/Users/Haghani/Desktop/mir_proj/mir_project/FastText_model.bin')



        self.get_embeddings()

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        for x in self.review_tokens:
            y = " ".join(x)
            self.embeddings.append(self.fasttext_model.get_query_embedding(y))
        
    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        label_encoder = LabelEncoder()
        x_train, x_test, y_train, y_test = train_test_split(self.embeddings, label_encoder.fit_transform(self.sentiments), test_size=test_data_ratio,
                                                            random_state=None)

        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

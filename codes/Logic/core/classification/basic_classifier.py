import numpy as np
from tqdm import tqdm
import sys
import os
import string


class BasicClassifier:
    def __init__(self):
        self.x = None

    def fit(self, x, y):
        self.x = None

    def predict(self, x):
        self.x = None

    def prediction_report(self, x, y):
        self.x = None

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """

        num_pos = 0

        for sentence in sentences:
            if ((self.predict([sentence])[0])=='positive'):
                num_pos += 1
        return num_pos / len(sentences)



import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import math
from collections import Counter


import sys
import os
import string
# Add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Add the directory containing indexes_enum.py to sys.path
basic_classifier_dir = os.path.join(project_root, 'Logic', 'core','classification','basic_classifier' )
sys.path.append(basic_classifier_dir)





from Logic.core.classification.basic_classifier import BasicClassifier


data_loader_dir = os.path.join(project_root, 'Logic', 'core','classification','data_loader' )
sys.path.append(data_loader_dir)






from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.x_train = x 
        self.y_train = y
        return self
    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        predictions = []  
        for x_test in tqdm(x):
            #neighbors = self.find_k_nearest_neighbors(x_test)
            distances = np.sqrt(np.sum((self.x_train - x_test) ** 2, axis=1))
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            prediction = Counter(k_nearest_labels).most_common(1)[0][0]

            predictions.append(prediction)

        return np.array(predictions)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        predictions = self.predict(x)

        return classification_report(y,predictions)


    def euclidean_distance(self,point_a, point_b):
 

        sum_squared_differences = 0
        for i in range(len(point_a)):
            difference = point_a[i] - point_b[i]
            sum_squared_differences += difference ** 2

        return math.sqrt(sum_squared_differences)


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader('C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/classification/IMDB_Dataset.csv')
    loader.load_data()

    X_train, X_test, y_train, y_test = loader.split_data()


    knn_classifier = KnnClassifier(n_neighbors=12)
    knn_classifier.fit(X_train, y_train)
    report = knn_classifier.prediction_report(knn_classifier.predict(X_test), y_test)
    print("knn classifier prediction report",report)


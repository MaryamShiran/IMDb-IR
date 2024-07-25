import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC



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






class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)


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
        y_pred = self.predict(x)
        return classification_report(y, y_pred)


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    
    loader = ReviewLoader('C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/classification/IMDB_Dataset.csv')
    loader.load_data()


    x_train,x_test, y_train,  y_test = loader.split_data()

    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)




    model = SVMClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as p
from sklearn.preprocessing import LabelEncoder 






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






preprocess_text_dir = os.path.join(project_root, 'Logic', 'core','word_embedding','fasttext_model' )
sys.path.append(preprocess_text_dir)






#from Logic.core.classification.data_loader import ReviewLoader








#from .basic_classifier import BasicClassifier
#from .data_loader import ReviewLoader
from Logic.core.word_embedding.fasttext_model import preprocess_text


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape

        self.prior = np.zeros(self.num_classes)
        for i, c in enumerate(self.classes):
            self.prior[i] = np.sum(y == c) / self.number_of_samples


        #self.embeddings = self.cv.fit_transform(x)

        #self.number_of_features = self.embeddings.shape[1]

        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))

        for i, class_label in enumerate(self.classes):
            class_data = x[y == class_label]
            feature_sums = class_data.sum(axis=0)
            total_class_sum = class_data.sum()
            smoothing_factor = self.alpha * self.number_of_features
    
            self.feature_probabilities[i, :] = (feature_sums + self.alpha) / (total_class_sum + smoothing_factor)


        self.log_probs = np.log(self.feature_probabilities)
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

    
        log_prior = np.log(self.prior)
        log_posterior = log_prior + np.dot(x, self.log_probs.T)
        return self.classes[np.argmax(log_posterior, axis=1)]





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
        report = classification_report(y, y_pred)
        return report


    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        x = self.cv.transform(sentences)
        predictions = self.predict(x)
        return np.sum(predictions == 1) / len(sentences)





# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    #loader = ReviewLoader('C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/classification/IMDB_Dataset.csv')

    df = p.read_csv('C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/classification/IMDB_Dataset.csv')
    reviews = [preprocess_text(text).split() for text in df['review']]

    sentiments = df['sentiment'].values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(sentiments)



#    loader = ReviewLoader('C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/classification/IMDB_Dataset.csv')
#    loader.load_data()


    cv = CountVectorizer(max_features=1000)


    reviews = [' '.join(review) if isinstance(review, list) else review for review in reviews]

    data=cv.fit_transform(reviews).toarray()

    x_train, x_test, y_train, y_test = train_test_split(data, sentiments, test_size=0.2, random_state=42)


    Myclassifier = NaiveBayes(cv)
    Myclassifier.fit(x_train, y_train)
    print(Myclassifier.prediction_report(x_test,(y_test)))

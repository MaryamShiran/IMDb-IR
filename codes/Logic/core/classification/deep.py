import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm











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













#from .data_loader import ReviewLoader
#from .basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        #self.device = 'mps' if torch.backends.mps.is_available else 'cpu'
        self.device = 'cpu'

        self.device = 'cuda' if torch.cuda.is_available() else self.device
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        #x = torch.FloatTensor(x)
        #y = torch.LongTensor(y)
        #train_dataset = ReviewDataSet(x, y)
        #self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        #best_val_f1 = 0
        #best_epoch = 0


        train_dataset = ReviewDataSet(x, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.to(self.device)


        for epoch in range(self.num_epochs):

            running_loss = 0.0

            self.model.train()

            for batch_idx, (embeddings, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")):
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(embeddings)
                    loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            tqdm.write(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss}")

            if self.test_loader:
                val_loss, val_f1 = self._eval_epoch(self.test_loader, self.model)
                tqdm.write(f"Validation Loss: {val_loss}, F1 Score: {val_f1}")

        self.best_model = self.model.state_dict()

        return self




    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.eval()
        test_dataset = ReviewDataSet(x, np.zeros(len(x)))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        labels = []
        with torch.no_grad():
            for embeddings, _ in test_loader:
                labels.extend(self.model(embeddings.to(self.device)).argmax(dim=1).cpu().tolist())


        return np.array(labels)


    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        model.eval()
        running_loss = 0.0
        predicted_labels = []
        true_labels = []


        with torch.no_grad():
            for embeddings, labels in dataloader:
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                output_labels = model(embeddings)

                running_loss += self.criterion(output_labels, labels).item()

                _, predicted = torch.max(output_labels, 1)

                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        f1_macro = f1_score(true_labels, predicted_labels, average='macro')

        average_loss = running_loss / len(dataloader)

        return average_loss, f1_macro



    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self
    

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        y_pred = self.predict(x)
        report = classification_report(y, y_pred, digits=4)
        return report
# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    review_loader = ReviewLoader('C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/classification/IMDB_Dataset.csv')
    review_loader.load_data()
    x_train, x_test, y_train, y_test = review_loader.split_data()


    classifier = DeepModelClassifier(in_features=x_train.shape[1], num_classes=len(np.unique(y_train)), batch_size=64,num_epochs = 4)

    classifier.fit(x_train, y_train)

    classifier.set_test_dataloader(x_test, y_test)


    print(classifier.prediction_report(x_test, y_test))

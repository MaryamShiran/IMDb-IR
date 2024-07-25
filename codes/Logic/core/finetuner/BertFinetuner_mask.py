#from transformers import AutoTokenizer, TFBertForSequenceClassification
#from datasets import Dataset
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import json
import torch
import collections
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import  BertTokenizerFast,BertForSequenceClassification
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from huggingface_hub import HfFolder, Repository
from torch.optim import AdamW



class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.dataframe = None
        self.data=None
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.top_n_genres)
        self.X_train=None
        self.X_val=None
        self.X_test = None
        self.y_train=None
        self.y_val=None
        self.y_test = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, "r") as f:
            data = json.load(f)

        self.dataframe =pd.DataFrame(data)


    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """



        genre_list = [genre for sublist in self.dataframe["genres"] for genre in sublist]
        genre_counter = Counter(genre_list)

        top_genres = {genre for genre, count in genre_counter.most_common(self.top_n_genres)}

        def filter_top_genres(genres, top_genres_set):
            return [genre for genre in genres if genre in top_genres_set]

        self.dataframe["genres"] = self.dataframe["genres"].apply(filter_top_genres, top_genres_set=top_genres)

        self.dataset = self.dataframe[self.dataframe["genres"].apply(len) > 0]


        exploded_genres = self.dataset['genres'].explode()

        genre_counts = exploded_genres.value_counts()

        plt.figure(figsize=(8, 8))
        plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Genre Distribution')
        plt.axis('equal')  
        plt.show()


        #print("done")

        return


    def split_dataset(self, test_size=0.2, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        X = self.dataset["first_page_summary"].tolist()
        y = self.dataset['genres'].str[0].tolist()


        y = LabelEncoder().fit_transform(y)

        self.X_train, X_temp, self.y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)


    def create_dataset(self, data__, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        #  Implement dataset creation logic


        data_ = list(map(str, data__))


        encodings = self.tokenizer(list(data_), truncation=True, padding=True)
        return IMDbDataset(encodings, labels)


    def fine_tune_bert(
        self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01
    ):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """

        self.train_dataset = self.create_dataset(self.X_train, self.y_train)
        self.val_dataset = self.create_dataset(self.X_val, self.y_val)
        self.test_dataset = self.create_dataset(self.X_test, self.y_test)



        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(
            output_dir='./results',
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        self.model = trainer.model


    def compute_metrics(self, preds, labels):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        res={}
        
        res["accuracy"] = metrics.accuracy_score(labels, preds)
        f1 = metrics.f1_score(labels, preds, average='macro')
        res["f1"]=f1

        return res

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # we here used batch evaluation but you can use single evaluation.
        # point is batch evaluation seems to be faster than single version


        test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        self.model.to(self.device)
        self.model.eval()

 
        predictions =[]
        true_labels = []


        for batch in test_loader:
            batch = {m: n.to(self.device) for m, n in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            labels = batch["labels"]


            preds = torch.argmax(logits, dim=1).flatten().detach().cpu().numpy()
            labels = labels.flatten().cpu().numpy()
            predictions.append(preds)
            true_labels.append(labels)




        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)


        metrics = self.compute_metrics(predictions, true_labels)

        print("Evaluation Metrics:")
        print(metrics)

    def save_model(self, model_name, to_huggingface=True):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        
        from huggingface_hub import login, create_repo

        #self.model.save_model(model_name)
        self.tokenizer.save_pretrained(model_name)

        token = "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"
        login(token)

        repo_url = create_repo(repo_id=model_name, private=True, exist_ok=True)


        self.model.push_to_hub(model_name, token)
        self.tokenizer.push_to_hub(model_name, token)


        print(f"Model '{model_name}' has been saved to Hugging Face Hub at {repo_url}")



class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.labels = labels
        self.data = encodings

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """


        item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)

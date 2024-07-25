import sys
import os
import string
# Add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Add the directory containing indexes_enum.py to sys.path
indexer_dir = os.path.join(project_root, 'Logic', 'core', 'indexer')
sys.path.append(indexer_dir)

from Logic.core.indexer.index_reader import Index_reader
from Logic.core.indexer.indexes_enum import Indexes


import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
#from ..indexer.index_reader import Index_reader
#from ..indexer.indexes_enum import Indexes
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        
        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        
        data = Index_reader(self.file_path, index_name=Indexes.DOCUMENTS).index
        

        titles = [movie_data.get('title', '') for movie_id, movie_data in tqdm(data.items())]
        genres = [movie_data.get('genres', '') for movie_id, movie_data in tqdm(data.items())]
        synopses = [movie_data.get('synopsis', '') for movie_id, movie_data in tqdm(data.items())]
        summaries = [movie_data.get('summaries', '') for movie_id, movie_data in tqdm(data.items())]
        reviews = [movie_data.get('reviews', '') for movie_id, movie_data in tqdm(data.items())]

#        print(reviews[0])
#        print(summaries[0])
#        print(genres[0])
#        print(titles[0])
#        print(synopses[0])


        df = pd.DataFrame({
            'synopsis': synopses,
            'summary': summaries,
            'reviews': reviews,
            'title': titles,
            'genre': genres
        })
        return df


    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        
        df = self.read_data_to_df()

        print(df)


        label_encoder = LabelEncoder()
        first_genres = []
        for genre in df['genre']:
            

            if len(genre) >=1:
                first_genres.append((genre.split())[0])
            else:
                first_genres.append('N/A')

        df['first_genres'] = first_genres
        df['encoded_genres'] = label_encoder.fit_transform(df['first_genres'])


        my_beautiful_list=[]

        for sth in df["reviews"]:
            my_beautiful_list.append(self.preprocess_text(sth))            



        df["preprocessed_reviews"] = my_beautiful_list

        X = (df['preprocessed_reviews'].values)[:1000]
        y = (df['encoded_genres'].values)[:1000]

        #print(len(X))
        #print(len(y))

        #print(X)
        #print(y)
        return X, y
    

    def preprocess_text(self, text):

        if text is None:
            return ''
        else:

            #print(type(text))
            translation_table = str.maketrans('', '', string.punctuation)
            text = text.lower()
            text = text.translate(translation_table)


            tokens = word_tokenize(text)

            stop_words = set(stopwords.words('english'))

            res=''
            for token in tokens:
                if token not in stop_words:
                    res=res+ token + " "

            return res

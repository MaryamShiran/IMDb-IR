import fasttext
import re
import tempfile
from tqdm import tqdm
import nltk
#nltk.download('punkt')
import numpy as np

import nltk
#nltk.download('stopwords')



import sys
import os
import string
# Add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Add the directory containing indexes_enum.py to sys.path
fasttext_loader_dir = os.path.join(project_root, 'Logic', 'core','word_embedding','fasttext_data_loader' )
sys.path.append(fasttext_loader_dir)






from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader



def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """


    translation_table = str.maketrans('', '', string.punctuation)
    
    if lower_case==True:
        text = text.lower()


    if punctuation_removal==True:
        text = text.translate(translation_table)


    tokens = word_tokenize(text)

    if stopword_removal==True:

        stop_words = set(stopwords.words('english'))
        if stopwords_domain:
            stop_words.add(stopwords_domain)
        
        tokens_new=[]
        for token in tokens:
            if token not in stop_words:
                tokens_new.append(token)
        tokens=tokens_new


    res=''
    for token in tokens:
        if len(token) >= minimum_length:
            res=res+token+' '

    return res


class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None


    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        with open(temp_file.name, 'w',encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        self.model = fasttext.train_unsupervised(input=temp_file.name, model=self.method)
        temp_file.close()


    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        return self.model.get_sentence_vector(query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        vec1 = self.model[word1]
        vec2 = self.model[word2]
        vec3 = self.model[word3]

        vec_res = vec2 - vec1 + vec3

        vector_words = {word: self.model[word] for word in self.model.words}
        excluding_words = {word1, word2, word3}

        min_distance= float('inf')
        closest = None
    
        filtered_dict = {key: value for key, value in vector_words.items() if key not in excluding_words}

        for word, emb_vec in filtered_dict.items():
            dist = distance.euclidean(vec_res, emb_vec)
            if dist < min_distance:
                closest = word
                min_distance = dist

        return closest


    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)


    def load_model(self, path="C:/Users/Haghani/Desktop/mir_proj/mir_project/FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)


    def prepare(self, dataset, mode, save=False, path="C:/Users/Haghani/Desktop/mir_proj/mir_project/FastText_model.bin"):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)

if __name__ == "__main__":
    fast_text_model = FastText( method='skipgram')


    path = "C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/indexer/index/"
    ft_data_loader = FastTextDataLoader(path)



    X , y = ft_data_loader.create_train_data()
    np.savez('arrays.npz', arr1=X, arr2=y)

    fast_text_model.train(X)

    print("number of words", len(fast_text_model.model.words))

    fast_text_model.prepare(None, mode = "save")
    fast_text_model.save_model()

    #print("number of words", len(fast_text_model.words))

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = fast_text_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "woman"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {fast_text_model.analogy(word1, word2, word3)}")

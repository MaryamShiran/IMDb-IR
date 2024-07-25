
#import nltk
#from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import json
import string
import os


class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        
        self.documents = documents
        self.stopwords = ["this","that","about","whom","being","where","why","had","should","each"] 

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.

        """
        all_docs=[]
        for doc in self.documents:
            preprocessed_doc={}
            for field in doc:

                

                y= str(doc[field])

                x=self.remove_links(y)
                x=self.remove_punctuations(x)
                x=self.tokenize(x)
                x=self.normalize(x)
                x=self.remove_stopwords(x)

                preprocessed_doc[field]=x


            all_docs.append(preprocessed_doc)

        return all_docs

    def normalize(self, words):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        tokens list
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        # Lowercasing
        words_lower = [word.lower() for word in words]

        # Stemming
        #stemmer = PorterStemmer()
        #words_stemmed = [stemmer.stem(word) for word in words_lower]

        # Lemmatization
        #lemmatizer = WordNetLemmatizer()
        #words_lemmatized = [lemmatizer.lemmatize(word) for word in words_lower]

        return words_lower

        

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """

        #print("hj",type(text))
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
    
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        #import string
        return ''.join(char for char in text if char not in string.punctuation)

        

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        tokens = [word for word in text.split()]
        return tokens

    def remove_stopwords(self, words):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        words : list
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        filtered_doc = ' '.join([word for word in words if word not in self.stopwords])
        return filtered_doc
        

if __name__ == "__main__":


    json_file_path = "C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/IMDB_crawled.json"


    with open(json_file_path, "r") as file:
        data = json.load(file)
    

    x=Preprocessor(data)


    preprocessed_data = x.preprocess()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_documents = os.path.join(script_dir, 'preprocessed_documents.json')




    with open(preprocessed_documents, 'w') as f:
        json.dump(list(preprocessed_data), f)




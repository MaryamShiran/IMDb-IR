import time
import os
import json
import copy
from indexes_enum import Indexes
import collections

class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.


        """

        #self.terms=[]

        self.preprocessed_documents = preprocessed_documents


        self.index = None

        self.Index()






    def Index(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = collections.defaultdict(lambda: collections.defaultdict(int))



        for document in self.preprocessed_documents:
                for star in (str(document['stars'])).split():
                    #self.terms.append(star)
                    #print(star)

                    current_index[star][document["id"]] += 1
                for genre in (str(document['genres'])).split():
                    #self.terms.append(genre)
                        #print(genre)
                    current_index[genre][document["id"]] += 1

                for word in (str(document['summaries'])).split():
                    #self.terms.append(word)
                #print(word)
                #for word in summary_part.split():
                    current_index[word][document["id"]] += 1
        
        self.index=current_index
        return current_index

 



 

    def store_index(self, path: str):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)



        file_path = os.path.join(path, "all_index.json")

        with open(file_path, 'w') as f:
            json.dump(self.index, f) 





if __name__ == "__main__":


    json_file_path = "C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/preprocessed_documents_standard.json"


    with open(json_file_path, "r") as file:
        data = json.load(file)


    #data=data[0:3]
    

    x=Index(data)
    #print(x)
    #x.index()

    x.store_index('./index')









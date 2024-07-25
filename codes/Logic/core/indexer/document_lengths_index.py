import json
from indexes_enum import Indexes,Index_types
from index_reader import Index_reader

class DocumentLengthsIndex:
    def __init__(self,path='index/'):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.

        """
        self.path="c:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/indexer/index/"

        self.documents_index = Index_reader(self.path, index_name=Indexes.DOCUMENTS).index

        self.document_length_index = {
            Indexes.STARS: self.get_documents_length(Indexes.STARS),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES),
            Indexes.SUMMARIES: self.get_documents_length(Indexes.SUMMARIES)
        }
        self.store_document_lengths_index(self.path, Indexes.STARS)
        self.store_document_lengths_index(self.path, Indexes.GENRES)
        self.store_document_lengths_index(self.path, Indexes.SUMMARIES)

    def get_documents_length(self, where):
        """
        Gets the documents' length for the specified field.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.

        Returns
        -------
        dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field (where).
        """




        ind = Index_reader(self.path,index_name=where).index


        #print(ind)

        



        dict = {} 

        for document_id in self.documents_index.keys():  

            s=0

            for term in ind.keys():
                if document_id in (ind[term]).keys():  
                    s=s+ (ind[term])[document_id]

            dict[document_id]=s


        return dict



        
    
    def store_document_lengths_index(self, path , index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        path = path + index_name.value + '_' + Index_types.DOCUMENT_LENGTH.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.document_length_index[index_name], file, indent=4)
    

if __name__ == '__main__':
    document_lengths_index = DocumentLengthsIndex()
    print('Document lengths index stored successfully.')
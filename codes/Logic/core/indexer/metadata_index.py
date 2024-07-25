from index_reader import Index_reader
from indexes_enum import Indexes, Index_types
import json

class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """

        path="c:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/indexer/index/"

        #self.all_Indexes={}

        #Index_names=[Indexes.STARS,Indexes.GENRES,Indexes.SUMMARIES]

    


        #for what in Index_names:
        #    x=Index_reader(path,what)
        #    self.all_Indexes[what]=x



        self.index = {
            Indexes.STARS: Index_reader(path, index_name=Indexes.STARS).index,
            Indexes.GENRES: Index_reader(path, index_name=Indexes.GENRES).index,
            Indexes.SUMMARIES: Index_reader(path, index_name=Indexes.SUMMARIES).index,
        }
        
        self.documents=self.read_documents()
        self.metadata_index=self.create_metadata_index()
        self.store_metadata_index(path)


        
        

    def read_documents(self):
        """
        Reads the documents.
        
        """

        json_file_path = "C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/preprocessed_documents_standard.json"


        with open(json_file_path, "r") as file:
            data = json.load(file)



        return data

    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['averge_document_length'] = {
            'stars': self.get_average_document_field_length(Indexes.STARS),
            'genres': self.get_average_document_field_length(Indexes.GENRES),
            'summaries': self.get_average_document_field_length(Indexes.SUMMARIES)
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    def get_average_document_field_length(self,where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """

        ind=self.index[where]

        #print(ind)

        s=0

        for term in ind.keys():
            term_list=ind[term]
            for y in term_list:
                s=s+term_list[y]

        return s/len(self.documents)


    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path =  path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


    
if __name__ == "__main__":
    meta_index = Metadata_index()
 #   meta_index.(store_metadata_index)("c:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/indexer/index/")

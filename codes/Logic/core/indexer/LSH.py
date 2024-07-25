import numpy as np
import itertools
import random
import json
import string



class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """



        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """



        words = document.split()
        shingles = []

        if k <= 0 or k > len(words):
            return shingles

        for i in range(len(words) - k + 1):
            k_tuple = " ".join(words[i:i + k])
            shingles.append(k_tuple)


        #print(shingles)

        return shingles




    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """

        all_shingles= set()
        for doc  in self.documents:
            shings = self.shingle_document(doc["summaries"])
            all_shingles.update(shings)

        #print(all_shingles)

        characteristic_matrix = np.zeros((len(all_shingles), len(self.documents)))

        for j, doc in enumerate(self.documents):
            shingles = self.shingle_document(doc["summaries"])
            for i, prop in enumerate(all_shingles):
                if prop in shingles:
                    characteristic_matrix[i, j] = 1


        #print("sd",characteristic_matrix)
 


        return characteristic_matrix

    def min_hash_signature(self,binary_matrix):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """


        #print(binary_matrix)



        #print("hjlk",type(binary_matrix))
        m, n = binary_matrix.shape
        #print(m,"ghk",n)
        T=self.num_hashes
        all_w = np.zeros((T, n))
    
        for t in range(T):
            permutation = np.random.permutation(m)
            w = np.zeros(n)


        
            for j in range(n):

                L = 0

   
                while True:


                    #print(L)
                    #print(permutation)


                    if L>=m:
                        w[j] = 0
                        break
                        

                    position = (np.where(permutation == L))[0][0]

      

                    if binary_matrix[position][j] == 1:
                        w[j] = position
                        break
                    L += 1




        
            all_w[t] = w

        return all_w

            

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.


        """

        signature=signature.astype(int)

        
        bucket_dict = {}

        for i in range(bands):
            for j in range(signature.shape[1]):
                s=""
                for l in range(rows_per_band):
                    s += (str(signature[i*rows_per_band+l][j]))
                #print(s)

                
                #binary_string = bin(i)[2:]
                #print(s)
                s += (str(i))
                #print(s)
                decimal_value = int(s)
                hash_value=0
                for digit in str(decimal_value):
                    hash_value = (hash_value * 31 + int(digit)) % (10**9 + 7)

                if hash_value in bucket_dict.keys():
                    bucket_dict[hash_value].append(j)

                else:
                    bucket_dict[hash_value] =[]
                    bucket_dict[hash_value].append(j)

        return bucket_dict





    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """

        x=self.build_characteristic_matrix()
        x=self.min_hash_signature(x)
        return self.lsh_buckets(x,10, (x.shape[0])//10)



        

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        first_set=set(first_set)
        second_set=set(second_set)
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))


        return 0 if union == 0 else intersection / union
        

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id]["summaries"], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id]["summaries"], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id]["summaries"], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


if __name__ == "__main__":


    json_file_path = "C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/LSHFakeData.json"


    with open(json_file_path, "r") as file:
        Fakedata = json.load(file)

    json_file_path = "C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/IMDB_crawled.json"


    with open(json_file_path, "r") as file:
        data = json.load(file)

    data=data[1:150]

    data.extend(Fakedata)


    docId_sumary=[]


    for doc in data:
        a={}
        a["id"]=doc["id"]
 #       summary=""



  #      for summary_part in doc["summaries"]:

  #          print(summary_part)

            
  #          summary=summary.join((summary_part))


        a["summaries"]=(("".join(doc["summaries"])).translate(str.maketrans("", "", "[]"+string.punctuation)))

        docId_sumary.append(a)








    x=MinHashLSH(docId_sumary,20)


    y=x.perform_lsh()
    x.jaccard_similarity_test(y,docId_sumary)








    





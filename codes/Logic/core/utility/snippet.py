import re
import string
class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        stop_words = ["this","that","about","whom","being","where","why","had","should","each"]  

        query_n = query.translate(str.maketrans('', '', string.punctuation))

        query_words = query_n.split()
        filtered_query = ' '.join([word for word in query_words if word.lower() not in stop_words])
        return filtered_query

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """


        final_snippet = ""
        not_exist_words = []


        doc=doc.lower()


        punctuations = string.punctuation
    
        cleaned_string = ''.join(char for char in doc if char not in punctuations)

        doc=cleaned_string


        
        tokened_doc=doc.split() #???

        #print(tokened_doc)

        tokened_doc1=tokened_doc

        f_query = self.remove_stop_words_from_query(query)

        f_query=f_query.lower()

        query_tokens = f_query.split()

        query_tokens_set=set(query_tokens)


        not_exist_words=query_tokens_set - set(tokened_doc)




        C = []
        for word in query_tokens_set:
            for i in range(len(tokened_doc1)):
                #print(tokened_doc1[i])
                #print(word)
                if tokened_doc1[i]==word:
                    C.append(i)
            #C.extend([i for i, w in enumerate(tokened_doc) if w == word])

        C = sorted(C)

        #print(C)


        num_set = set(C)
        result = set()
        for num in num_set:
            for i in range(max(0,num - self.number_of_words_on_each_side),min( len(tokened_doc1),num + self.number_of_words_on_each_side + 1)):
                
                result.add(i)

        result=list(result)

        final_snippet=""

        #print("res",result)

        for i in range (len(result)):
            #print(final_snippet)
            res=result[i]

            #print(res)
            
            if res in C:
                
                final_snippet=final_snippet+" ***"+(tokened_doc[res])+"***"


            else:
                final_snippet=final_snippet+" "+tokened_doc[res]
            
            if (i != len(result) - 1) and result[i+1] > (result[i] + self.number_of_words_on_each_side):
                final_snippet=final_snippet+"..."

        return final_snippet, not_exist_words
    

if __name__ == "__main__":
    s=Snippet(5)
    y,z=s.find_snippet("Captain Jack Sparrow is pursued by old rival Captain Salazar and a crew of deadly ghosts who have escaped from the Devil's Triangle. They're determined to kill every pirate at sea...notably Jack.","jack")
    print(y)

import numpy as np
import math

class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self,query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            df=self.get_list_of_documents([term])

            self.idf[term]=math.log((self.N + 0.1) / (len(df)+0.1))
            
        return self.idf[term]
    
    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """


        query_tfs = {}

        for word in query:
            query_tfs[word] = query_tfs.get(word, 0) + 1

        return query_tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """



        docs = self.get_list_of_documents(query)

        query_tfs=self.get_query_tfs(query)


        scores={}


        for doc in docs:
            scores[doc]=self.get_vector_space_model_score(query, query_tfs,doc, method[4:7] , method[0:3])

        return scores 
           


    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        q1=query_method[0]
        q2=query_method[1]
        q3=query_method[2]


        d1=document_method[0]
        d2=document_method[1]
        d3=document_method[2]



        q1final=query_tfs

        q2final = {item: 1 for item in query_tfs.keys()}


        if q1=='l':
            for word in query_tfs.keys():
                q1final[word]=1+math.log(query_tfs[word])
        
        if q2=='t':
            for word in q2final.keys():
                q2final[word]=self.get_idf(word)



        
        result_dict = {key: q1final[key] * q2final  [key] for key in q1final}

        if q3=='c':

            s=0
            
            for key in result_dict.keys():
                s=s+((result_dict[key])**2)

            s=math.sqrt(s)


            for word in result_dict.keys():
                result_dict[word]=(result_dict[word])/s


        













        d1final={}
        
    #    for term in query_tfs.keys:
    #        d1final[term]=self.index[term][document_id]



        d1final = {}  # Dictionary to store word frequencies in the document
  
        for term in self.index.keys():
            if document_id in self.index[term].keys():  
                d1final[term] = self.index[term][document_id]




        d2final = {item: 1 for item in d1final.keys()}


        if d1=='l':
            for word in d1final.keys():
                d1final[word]=1+math.log(d1final[word])
        
        if d2=='t':
            for word in d2final.keys():
                d2final[word]=self.get_idf(word)



        
        result_dict_d = {key: d1final[key] * d2final  [key] for key in d1final}

        if d3=='c':

            s=0
            
            for key in result_dict_d.keys():
                s=s+((result_dict_d[key])**2)

            s=math.sqrt(s)


            for word in result_dict_d.keys():
                result_dict_d[word]=(result_dict_d[word])/s


        common_keys = set(result_dict.keys()) & set(result_dict_d.keys())  
        result = sum(result_dict[key] * result_dict_d[key] for key in common_keys) 

        return result




    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """


        docs = self.get_list_of_documents(query)



        scores={}


        for doc in docs:
            scores[doc]=self.get_okapi_bm25_score( query, doc, average_document_field_length, document_lengths)
        return scores 
        

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        k1=2
        b=0.75

        query_tfs=self.get_query_tfs(query)

        s=0

        for word in query_tfs.keys():

            tft=0
            if word in self.index.keys() and document_id in self.index[word]:
                tft=self.index[word][document_id]


            

            s=s+ self.get_idf(word) * (((k1+1)*tft) /((k1*(1-b+b*(document_lengths[document_id]/average_document_field_length)))+(tft)))


        return s



    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """

        word_prior_probs = {}
        N=0
        
        for word in self.index.keys():
            for doc in self.index[word].keys():
                N = N+self.index[word][doc]

        U= len(self.index.keys())


        query_tfs=self.get_query_tfs(query)


        for word in query_tfs.keys():

            t_cm=0

            if word in self.index.keys():
                for doc in self.index[word].keys():
                    t_cm = t_cm+self.index[word][doc]

            word_prior_probs[word]=t_cm/N
                #M_c=0
                #for d in self.index[word]:
        










        docs = self.get_list_of_documents(query)



        scores={}


        for doc in docs:
            scores[doc]=self.compute_score_with_unigram_model( query, doc, smoothing_method, document_lengths,alpha, lamda,word_prior_probs,U)
        return scores 

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda,word_prior_probs,U
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """



        if smoothing_method=="bayes":


 

            query_tfs=self.get_query_tfs(query)

            prob=1


            for word in query_tfs.keys():

                tft=0

                if word in self.index.keys() and document_id in self.index[word]:
                    tft=self.index[word][document_id]
                    #M_c=0
                    #for d in self.index[word]:
                        

                    prob=prob*(((tft+alpha*(word_prior_probs[word]))/(document_lengths[document_id]+alpha))**tft)

            return prob
                


 

        if smoothing_method=="naive":


 

            query_tfs=self.get_query_tfs(query)

            prob=1


            for word in query_tfs.keys():

                tft=0

                if word in self.index.keys() and document_id in self.index[word]:
                    tft=self.index[word][document_id]
                    #M_c=0
                    #for d in self.index[word]:
                
        
                        

                prob=prob*((((tft+(1/U)))/(document_lengths[document_id]+1))**tft)

            return prob
    

        if smoothing_method=="mixture":


 

            query_tfs=self.get_query_tfs(query)

            prob=1


            for word in query_tfs.keys():

                tft=0

                if word in self.index.keys() and document_id in self.index[word]:
                    tft=self.index[word][document_id]
                    #M_c=0
                    #for d in self.index[word]:
                
        
                        

                prob=prob*((lamda*(tft/(document_lengths[document_id]))+(1-lamda)*(word_prior_probs[word]))**tft)

        return prob                           





        
        
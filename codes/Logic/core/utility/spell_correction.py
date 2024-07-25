import string
class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        word = '$' + word + '$'

        for i in range(len(word) - k +1):
            shingle = word[i:i+k]
            shingles.add(shingle)

        return shingles


      

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))


        return 0 if union == 0 else intersection / union
    

        
    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """


        all_shingled_words = {}
        word_counter = {}

  
        for  word in all_documents:
           
            if word in word_counter.keys():
                word_counter[word]= word_counter[word]+1

            else:
                word_counter[word]= 1
                all_shingled_words[word] = self.shingle_word(word)
                  


        return all_shingled_words, word_counter

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """

        #print(self.word_counter)
        #top5_candidates = list()


        set1=self.shingle_word(word)


        top_scores = []
        
        for Cword in self.word_counter.keys():
            set2=self.all_shingled_words[Cword]
            #print(set2)
            #break
            x=self.jaccard_score(set1,set2)
            #print(x)
            #break

            if len(top_scores) < 5:
                top_scores.append((Cword, x))
                top_scores.sort(key=lambda x: x[1], reverse=True)
                #print(top_scores)
                #break
            else:
                if x > top_scores[4][1]:
                    top_scores.pop()
                    top_scores.append((Cword, x))
                    top_scores.sort(key=lambda x: x[1], reverse=True)
                    #print(top_scores)


                    #break
        #print(top_scores)

        return top_scores



    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""


        query_sentence_no_punctuation = query.translate(str.maketrans('', '', string.punctuation))

        for word in (str(query_sentence_no_punctuation)).split():

            word=word.lower()


            x=self.find_nearest_words(word)
            #print(x)

            l=[]

            for xx in x:
                l.append(self.word_counter[xx[0]])

            m=max(l)






        
            
            #m = max([(self.word_counter[w[0]]) for w in x])
            #print(m)
            pre_result = [(a, a_score * self.word_counter[a] / m) for a, a_score in x]

            print("pre_result",pre_result)



            max_score_element = max(pre_result, key=lambda x: x[1])[0]

            print("max_score_element",max_score_element)



            final_result = final_result+(max_score_element)+(" ")


        return final_result
    
#if __name__ == "__main__":
#    with open("C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/terms.json", 'r') as file:

#        movies_dataset=[line.strip() for line in file]

#    s=SpellCorrection(movies_dataset)


#    a=s.spell_check("JoKwr hrenry")

    #print("a",a)
    

    
import sys
import os
import string
# Add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Add the directory containing indexes_enum.py to sys.path
search_dir = os.path.join(project_root, 'Logic', 'core','search' )
sys.path.append(search_dir)





from typing import Dict, List



from Logic.core.search import SearchEngine

SpellCorrection_dir = os.path.join(project_root, 'Logic', 'core','utility','spell_correction' )
sys.path.append(SpellCorrection_dir)


from Logic.core.utility.spell_correction import SpellCorrection
#from core.snippet import Snippet

Index_dir = os.path.join(project_root, 'Logic', 'core','indexer','indexes_enum' )
sys.path.append(Index_dir)

from Logic.core.indexer.indexes_enum import Indexes, Index_types
import json
import string



with open("C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/terms.json", 'r',encoding='utf-8') as file:
    terms=[line.strip() for line in file]
#    print(terms[1:10])

with open("C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/IMDB_crawled_Standard.json", 'r') as f:
    movies_dataset = json.load(f)
#print(movies_dataset[0])

search_engine = SearchEngine()


def correct_text(text: str, all_documents: List[str]) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of str
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
    #preprocessor = Preprocessor([text])
    #query = preprocessor.preprocess()[0].split()


    
    #translator = str.maketrans('', '', string.punctuation)
    
    #cleaned_text = text.translate(translator).lower()
    

    spell_correction_obj = SpellCorrection(all_documents)
    text = spell_correction_obj.spell_check(text)
    return text


def search(
    query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weights: list = [0.3, 0.3, 0.4],
    unigram_smoothing=None,
    alpha=None,
    lamda=None,
    should_print=False,
    preferred_genre: str = None,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    mode: 'detailed' for searching in title and text separately.
          'overall' for all words, and weighted by where the word appears on.

    where: when mode ='detailed', when we want search query
            in title or text not both of them at the same time.

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    preferred_genre: A list containing preference rates for each genre. If None, the preference rates are equal.

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """



    new_weights={}

    #A=[]

    new_weights[Indexes.STARS]=weights[0]
    new_weights[Indexes.GENRES]=weights[1]
    new_weights[Indexes.SUMMARIES]=weights[2]

   

    weights = new_weights
    return search_engine.search(
        query, method, weights, max_results=max_result_count, safe_ranking=True, smoothing_method=unigram_smoothing, alpha=alpha, lamda=lamda
    )



def get_movie_by_id(id: str, movies_dataset):
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """

    #print(id)
    #print(len(movies_dataset))
    result = {}
    for movie in movies_dataset:
        if movie.get("id") == id:
            result = movie
    result["Image_URL"] = (
        "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"  # a default picture for selected movies
    )
    #print("this is the result",result )
    result["URL"] = (
        f"https://www.imdb.com/title/{result['id']}"  # The url pattern of IMDb movies
    )
    return result
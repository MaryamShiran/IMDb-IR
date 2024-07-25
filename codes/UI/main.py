import streamlit as st
import sys
import sys
import os
import string
import requests
from bs4 import BeautifulSoup
import numpy as np
import speech_recognition as sr
import pyaudio




# Add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)


sys.path.append("../")


link_dir = os.path.join(project_root, 'Logic', 'core','link_analysis','analyzer' )
sys.path.append(link_dir)

from Logic.core.link_analysis.analyzer import LinkAnalyzer



utils_dir = os.path.join(project_root, 'Logic', 'utils')
sys.path.append(utils_dir)



import Logic.utils as utils
import time
from enum import Enum
import random




TopRecent_dir = os.path.join(project_root, 'UI', 'TopRecent' )
sys.path.append(TopRecent_dir)

from UI.TopRecent import movies_list





top_recent_movies=movies_list






class Color(Enum):
    PRIMARY = "#4e79a7"
    SECONDARY = "#f28e2c"
    SUCCESS = "#76b7b2"
    INFO = "#59a14f"
    WARNING = "#edc948"
    DANGER = "#e15759"
    LIGHT = "#f7f7f7"
    DARK = "#343a40"
    TEXT_LIGHT = "#f0f2f6"  # Light background color
    TEXT_DARK = "#000000"   # Dark text color





Snippet_dir = os.path.join(project_root, 'Logic', 'core','utility','snippet')
sys.path.append(Snippet_dir)



from Logic.core.utility.snippet import Snippet



indexer_dir = os.path.join(project_root, 'Logic', 'core', 'indexer')
sys.path.append(indexer_dir)

from Logic.core.indexer.index_reader import Index_reader, Indexes



snippet_obj = Snippet()


class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"

headers = {
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
def get_IMG_url(url):

    session = requests.Session()
    session.headers.update(headers)

    try:
        r = session.get(url)
        
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, "html.parser")
            
            try:
                # Inspect and find the right element for the image
                poster_div = soup.find('div', class_='ipc-media__img')
                if poster_div:
                    image_tag = poster_div.find('img')
                    if image_tag and 'src' in image_tag.attrs:
                        image_url = image_tag['src']
                    else:
                        image_url = np.nan
                        print("Image tag not found or missing 'src' attribute.")
                else:
                    image_url = np.nan
                    print("Poster div not found.")
            except Exception as e:
                image_url = np.nan
                print(f"An error occurred: {e}")
        else:
            image_url = np.nan
            print(f"Failed to retrieve the webpage. Status code: {r.status_code}")
    except requests.exceptions.RequestException as e:
        image_url = np.nan
        print(f"Request failed: {e}")

    return image_url





def voice_search():
    print("1")
    r = sr.Recognizer()
    print("r")
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio)
        st.write(f"You said: {query}")
        return query
    except sr.UnknownValueError:
        st.warning("Sorry, could not understand your audio.")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""












def display_top_recent_movies(movies):
    st.markdown("## Most Popular Movies")
    for movie in movies:
        card = st.columns([3, 1])
        with card[0]:
            st.title(movie["title"])
            st.markdown(f"**Rating:** {movie['rating']}")
            #st.markdown(f"**Genres:** {movie['genre']}")
            st.markdown(f"**Year:** {movie['year']}")
            #st.markdown(f"**Release Date:** {movie['release_date']}")
            st.markdown("**Directors:**")
            for actor in movie['director'].replace(',', '').split():
                st.text(actor)

            st.markdown("**Genres:**")
            for gnr in movie['genre'].replace(',', '').split():
                st.text(gnr)




            #st.markdown("**Directors:**")
            #st.text(movie["director"])

        with card[1].container():
            movie["Image_URL"]=get_IMG_url(movie["IMDb URL"])
            st.image(movie["Image_URL"], use_column_width=True)




        st.divider()



def get_top_x_movies_by_rank(x: int, results: list):
    path = "../Logic/core/index/"  # Link to the index folder
    document_index = Index_reader(path, Indexes.DOCUMENTS)
    corpus = []
    root_set = []
    for movie_id, movie_detail in document_index.index.items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index.index[movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})

    #print(root_set)
    
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies


def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary


def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))


def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    unigram_smoothing,
    alpha,
    lamda,
    filter_button,
    num_filter_results,
    display_mode,
    text_mode,
    max_num
):
    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            st.markdown(f"**Top {num_filter_results} Actors:**")
            actors_ = ", ".join(top_actors)
            st.markdown(
                f"<span style='color:{random.choice(list(color)).value}'>{actors_}</span>",
                unsafe_allow_html=True,
            )
            st.divider()

        mask=max_num

        st.markdown(f"**Top {num_filter_results} Movies:**")
        for i in range(len(top_movies)):
            if mask==0:

                break
            else:
                mask=mask-1

            card = st.columns([3, 1])
            info = utils.get_movie_by_id(top_movies[i], utils.movies_dataset)

            if display_mode == "full":

                with card[0].container():
                    st.title(info["title"])
                    st.markdown(f"[Link to movie]({info['URL']})")
                    st.markdown(
                        f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                        unsafe_allow_html=True,
                    )

                with st.container():
                    st.markdown("**Directors:**")

                    if info["directors"] is not None:

                        num_authors = len(info["directors"])
                        for j in range(num_authors):
                            st.text(info["directors"][j])

                    else:
                        st.text("No directors information available.")




                with st.container():
                    st.markdown("**Stars:**")
                    if info["stars"] is not None:

                        num_authors = len(info["stars"])
                        stars = "".join(star + ", " for star in info["stars"])
                        st.text(stars[:-2])


                        topic_card = st.columns(1)
                        with topic_card[0].container():
                            st.write("Genres:")
                            num_topics = len(info["genres"])
                            for j in range(num_topics):
                                st.markdown(
                                    f"<span style='color:{random.choice(list(color)).value}'>{info['genres'][j]}</span>",
                                    unsafe_allow_html=True,
                                )





                    else:
                        st.text("No stars information available.")




                with card[1].container():
                    info["Image_URL"]=get_IMG_url(info['URL'])

                    st.image(info["Image_URL"], use_column_width=True)

                #st.divider()

            elif display_mode == "compact":
                with card[0].container():
                    st.title(info["title"])
                    genres = ", ".join(info["genres"])
                    st.write(f"Genres: {genres}")

            st.divider()





        return
    

    if search_button:
        corrected_query = utils.correct_text(search_term, utils.terms)

        #print(search_term)
        #print(corrected_query)



        if (corrected_query.strip() != search_term.strip()):
            if text_mode=="Yes":

                st.warning(f"Your search terms were corrected to: {corrected_query}")
                search_term = corrected_query
            else:
                st.warning(f"Below are results for your searched query, however, you may like to consider correcting it to: {corrected_query}")
  

        gif_placeholder = st.empty()

        with st.spinner("Searching..."):
            #gif_placeholder = st.markdown(gif_html, unsafe_allow_html=True)


            gif_html = """
            <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
                <iframe src="https://giphy.com/embed/KxscqRylVTBty6bUIS" width="480" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
            </div>
            <p style="text-align: center;"><a href="https://giphy.com/gifs/Stolengoods-animation-motiongraphics-stolen-goods-KxscqRylVTBty6bUIS">via GIPHY</a></p>
            """

            # Use the placeholder to display the centered GIF
            gif_placeholder.markdown(gif_html, unsafe_allow_html=True)



            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
                unigram_smoothing=unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
            )
            gif_placeholder.empty()

            if "search_results" in st.session_state:
                st.session_state["search_results"] = result
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

        
        #print(result[1][0])     #usless and might even cuse error when the result has only one item.



        mask=max_num
        for i in range(len(result)):
            if mask==0:

                break
            else:
                mask=mask-1
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
            if display_mode == "full":

                with card[0].container():
                    st.title(info["title"])
                    st.markdown(f"[Link to movie]({info['URL']})")
                    st.write(f"Relevance Score: {result[i][1]}")
                    st.markdown(
                        f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                        unsafe_allow_html=True,
                    )

                with st.container():
                    st.markdown("**Directors:**")
                    if info["directors"] is not None:

                        num_authors = len(info["directors"])
                        for j in range(num_authors):
                            st.text(info["directors"][j])

                    else:
                        st.text("No directors information available.")





                with st.container():
                    st.markdown("**Stars:**")
                    if info["stars"] is not None:

                        num_authors = len(info["stars"])
                        stars = "".join(star + ", " for star in info["stars"])
                        st.text(stars[:-2])

                    else:
                        st.text("No stars information available.")




                    topic_card = st.columns(1)
                    with topic_card[0].container():
                        st.write("Genres:")
                        num_topics = len(info["genres"])
                        for j in range(num_topics):
                            st.markdown(
                                f"<span style='color:{random.choice(list(color)).value}'>{info['genres'][j]}</span>",
                                unsafe_allow_html=True,
                            )
                with card[1].container():
                    info["Image_URL"]=get_IMG_url(info['URL'])
                    st.image(info["Image_URL"], use_column_width=True)

                #st.divider()





            elif display_mode == "compact":
                with card[0].container():
                    st.title(info["title"])
                    genres = ", ".join(info["genres"])
                    st.write(f"Genres: {genres}")

            st.divider()








        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                "search_results" in st.session_state
                and len(st.session_state["search_results"]) > 0
            )





def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://wallpapercave.com/wp/wp9672819.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )




def main():

#    st.title("Top Recent Movies")
#    st.write(
#        "These are top recent movies, scroll down to search for the movie you like."
#    )







    st.set_page_config(page_title="IMDb Search Engine", page_icon="🎬", layout="wide")


          #  background: url('C:/Users/Haghani/Desktop/mir_proj/mir_project/UI/wp8923985.webp') no-repeat center center fixed;    





    set_bg_hack_url()
    st.title("🎥 IMDb Movie Search Engine")
    set_bg_hack_url()

    st.write(
        "This is a simple search engine for IMDB movies. You can search through IMDB dataset and find the most relevant movie to your search terms."
    )
    st.markdown(
        '<span style="color:yellow">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input("Seacrh Term")

#    if st.button("Text Search"):
        # Perform search based on text input
#        st.write(f"Searching for: {search_term}")

#    if st.button("Voice Search"):
        # Perform voice search
#        query = voice_search()
#        if query:
#            st.write(f"Performing search for: {query}")
            # Implement your search logic here using the 'query' variable

    with st.sidebar:
        st.header("Settings")
        num_results=st.slider("Max number of results", 1, 100, 10)

 



    with st.sidebar:

        st.header("Browse")

        browse_top_movies_button = st.button("Browse In Most Popular Movies")








        st.header("Advanced Search Options")
        # Include advanced search inputs here



        with st.expander("Advanced Search"):
            search_max_num = st.number_input(
                "Maximum number of results", min_value=5, max_value=100, value=10, step=5
            )
            weight_stars = st.slider(
                "Weight of stars in search",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )

            weight_genres = st.slider(
                "Weight of genres in search",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )

            weight_summary = st.slider(
                "Weight of summary in search",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )
            slider_ = st.slider("Select the number of top movies to show", 1, 10, 5)

            search_weights = [weight_stars, weight_genres, weight_summary]
            search_method = st.selectbox(
                "Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram")
            )

        unigram_smoothing = None
        alpha, lamda = None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox(
                "Smoothing method",
                ("naive", "bayes", "mixture"),
            )
            if unigram_smoothing == "bayes":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            if unigram_smoothing == "mixture":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
                lamda = st.slider(
                    "Lambda",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )

        with st.sidebar:



            display_mode = st.radio("Display Mode", ["full", "compact"], index=0)
            text_mode = st.radio("Autocorrect Query", ["Yes", "No"], index=0)
    
        st.header("Rate us")


        with st.form("my_form"):
            st.text_input("Enter your name")
            st.slider("Rate this app", 1, 5)
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write("Form submitted")



    
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    search_button = st.button("Search!")
    filter_button = st.button("Filter movies by ranking")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        slider_,
        display_mode,
        text_mode,
        num_results
       
    )

    if browse_top_movies_button:
        display_top_recent_movies(top_recent_movies)



if __name__ == "__main__":
    main()











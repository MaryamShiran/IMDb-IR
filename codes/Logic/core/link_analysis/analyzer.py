import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Add the directory containing indexes_enum.py to sys.path
search_dir = os.path.join(project_root, 'Logic', 'core','link_analysis','graph' )
sys.path.append(search_dir)

from Logic.core.link_analysis.graph import LinkGraph
import networkx as nx
import json

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = set()
        self.authorities = set()
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            movie_id = movie["id"]
            self.hubs.add(movie_id)

            self.graph.add_node(movie_id)
            for star in movie["stars"].split():
                if star not in self.graph.graph.nodes():
                    self.graph.add_node(star)
                self.graph.add_edge(movie_id, star)

                self.authorities.add(star)





    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            movie_id = movie["id"]
            if movie["stars"] is not None:
                #print(type(movie["stars"]))
                #print(movie["stars"][0])
                for star in movie["stars"].split():
                    if any(star in root_movie["stars"].split() for root_movie in self.root_set):

                        #if movie_id not in self.graph.graph.nodes():
                        #    self.graph.add_node(movie_id)

                        if star not in self.graph.graph.nodes():
                            self.graph.add_node(star)

                        if movie_id not in self.graph.graph.nodes():
                            self.graph.add_node(movie_id)

                        self.graph.add_edge(movie_id, star)


                        self.authorities.add(star)
                        self.hubs.add(movie_id)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = []
        h_s = []

        print("I began")

        h, a = nx.hits(self.graph.graph, max_iter=num_iteration)

        print("not done")



        sorted_hubs = sorted(h, key=h.get, reverse=True)[:max_result]

        sorted_authorities = sorted(a, key=a.get, reverse=True)[:max_result]


        while len(h_s) < max_result:

            for s_h in sorted_hubs:

                #print("I am stuck in this while 1")
                
                if s_h.startswith('tt'):
                    h_s.append(s_h)



        while len(a_s)< max_result:

            for s_a in sorted_authorities:

                #print("I am stuck in this while 2")

                if not s_a.startswith('tt'):
                    a_s.append(s_a)


        return a_s, h_s


if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer

    with open("C:/Users/Haghani/Desktop/mir_proj/mir_project/Logic/core/IMDB_crawled_Standard.json", 'r') as f:
        corpus = json.load(f)

    root_set = (corpus[:50])   # TODO: it shoud be a subset of your corpus

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    print("expand_graph is done")
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')

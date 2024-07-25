import requests 
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
from queue import Queue
import json
import re
import os




class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'Accept-Language': 'en-US,en;q=0.9',

        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'

    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1200):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """

        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = []
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """

        return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """

        script_dir = os.path.dirname(os.path.abspath(__file__))
        crawled_file_path = os.path.join(script_dir, 'IMDB_crawled.json')
        not_crawled_file_path = os.path.join(script_dir, 'IMDB_not_crawled.json')



        with open(crawled_file_path, 'w') as f:
            json.dump(list(self.crawled), f)

        with open(not_crawled_file_path, 'w') as f:
            json.dump(list(self.not_crawled), f)


    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """

        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = set(json.load(f))

        with open('IMDB_not_crawled.json', 'w') as f:
            self.not_crawled = deque(json.load(f))


        for url in self.crawled:
          
          self.added_ids.add(self.get_id_from_URL(url))


      
        for url in self.not_crawled:
          
          self.added_ids.add(self.get_id_from_URL(url))

           

        
        
    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """

        x=requests.get(URL,headers=self.headers)





        return x


    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
 

        response = self.crawl("https://www.imdb.com/chart/top/")
        

        if response.status_code == 200:
 
            soup = BeautifulSoup(response.content, 'html.parser')
            movie_links = soup.find_all('a')



            for link in movie_links:
                href = link.get('href')
                if href and href.startswith('/title/'):
                    movie_id = self.get_id_from_URL("https://www.imdb.com" + href)
       
                    if movie_id not in self.added_ids:
 
                        self.not_crawled.append('https://www.imdb.com/title/' + movie_id + '/')
                        self.added_ids.add(movie_id)



    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO:
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        self.extract_top_250()
        # help variables

        #print(self.not_crawled)
        
        #print(self.crawled)



       # WHILE_LOOP_CONSTRAINTS = (len(self.crawled) < self.crawling_threshold)
        NEW_URL=None
        THERE_IS_NOTHING_TO_CRAWL = ( len(self.not_crawled)==0)

        


        #print("this is the new url",NEW_URL)


     
        futures = []
        crawled_counter = 0 

        with ThreadPoolExecutor(max_workers=20) as executor:
            while (crawled_counter < self.crawling_threshold):
                print("crawled_counter",crawled_counter)
                print("len(self.crawled)",len(self.crawled))

              #  print("self.crawling_threshold",self.crawling_threshold)





                if not (len(self.not_crawled)==0):
                    NEW_URL =  self.not_crawled.popleft()


                
                #print("in the while loop self.crawled is ",(self.crawled))
                URL = NEW_URL
                crawled_counter=crawled_counter+1
                futures.append(executor.submit(self.crawl_page_info, URL))
                if len(self.not_crawled)==0:
                    wait(futures)
                    futures = []

        #print(self.crawled)

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.

        Parameters
        ----------
        URL: str
            The URL of the site
        """
        #print("this is url",URL)
        print("new iteration")
        

        response = self.crawl(URL)
        #print("this is the response",response)
        #print("response code",response.status_code )
        if response.status_code == 200:
            #print("happy new year")
       
        #    print("len(self.crawled)",len(self.crawled))
        #    print((self.crawled))
            


            soup = BeautifulSoup(response.text, 'html.parser')
            #movie_links = soup.find_all('a')

            #print(movie_links)

            movie = self.get_imdb_instance()


            self.crawled.append(movie)


            #print("jjjjjjjjjj")

            self.extract_movie_info(response, movie, URL)


            #print(movie)

            #print(movie)

           
            #print("movie_links",len(movie_links))


            for link in movie['related_links']:
                #href = link.get('href')
               
                movie_id = self.get_id_from_URL(link)
                    #print(movie_id)
                if movie_id not in self.added_ids:
                        #print("new found",movie_id)
                    self.not_crawled.append(link)
                    self.added_ids.add(movie_id)










            #soup = BeautifulSoup(response.text, 'html.parser')
            #movie = self.get_imdb_instance()

        #    print(";l;;;ll;l;;",URL)
        #    self.crawled.add(URL)
        #    self.extract_movie_info(response, movie, URL)
        #    print("hjafkhlj",URL)


        #    related_links = self.get_related_links(soup)
        #    print("related links",related_links)
#            with self.add_queue_lock:
        #    for link in related_links:
        #        movie_id = self.get_id_from_URL(link)
        #        if movie_id not in self.added_ids:
        #            self.not_crawled.append('https://www.imdb.com/title/' + movie_id + '/')
        #            self.added_ids.add(movie_id)

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        #chrome_options = Options()
        #chrome_options.add_argument("--headless")  # Run in headless mode
        #chrome_options.add_argument("--disable-gpu")
        #driver = webdriver.Chrome(options=chrome_options)
        
        # Open the URL in the browser
        #driver.get(URL)
        
        # Wait for the page to load completely
        #WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
  




        try:
            soup = BeautifulSoup(res.text, 'html.parser')
            movie['id']=(self.get_id_from_URL(URL))
            #print("id",type(movie['id']))
            
            movie['title'] = self.get_title(soup)
            #print("title",type(movie['title']))
   
            movie['release_year'] = self.get_release_year(soup)
            #print("release_year",type(movie['release_year']))

            movie['mpaa'] = self.get_mpaa(soup)
            #print("mpaa",type(movie['mpaa']))

            movie['budget'] = self.get_budget(soup,URL)
            #print("budget",type(movie['budget']))

            movie['gross_worldwide'] = self.get_gross_worldwide(soup)
            #print("gross_worldwide",type(movie['gross_worldwide']))

            
            movie['directors'] = self.get_director(soup)
            #print("directors",type(movie['directors']))




            movie['writers'] = self.get_writers(soup)
            #print("writers",type(movie['writers']))

            movie['stars'] = self.get_stars(soup)
            #print("stars",type(movie['stars']))





            movie['related_links'] = self.get_related_links(soup)
            #print("related_links",type(movie['related_links']))





            movie['genres'] = self.get_genres(soup)
            #print("genres",type(movie['genres']))




            movie['languages'] = self.get_languages(soup)
            #print("languages",type(movie['languages']))


            movie['countries_of_origin'] = self.get_countries_of_origin(soup)
            #print("countries_of_origin",type(movie['countries_of_origin']))



            movie['rating'] = self.get_rating(soup)
            #print("rating",type(movie['rating']))



            movie['summaries'] = self.get_summary(URL)
            #print("summaries",type(movie['summaries']))


            movie['first_page_summary'] = movie['summaries'][0]
            #print("first_page_summary",type(movie['first_page_summary']))



            movie['synopsis'] = self.get_synopsis(URL)
            #print("synopsis",type(movie['synopsis']))





            movie['reviews'] = self.get_reviews_with_scores(soup, URL)
            #print("reviews",type(movie['reviews']))

        except Exception as e:
            print("An error occurred:", str(e))

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            movie_id = url.split('/')[-2]
            summary_link = f"https://www.imdb.com/title/{movie_id}/plotsummary/"
            return summary_link
        except Exception as e:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            movie_id = url.split('/')[-2]
            review_link = f"https://www.imdb.com/title/{movie_id}/reviews/"
            return review_link
        except Exception as e:
            print("failed to get review link")

    def get_title(self, soup):
        
        try:
            title_tag = soup.find('h1', attrs={'data-testid': 'hero__pageTitle'})  
            
            if not title_tag:
                title_tag = soup.find('h1', class_='hero_primary-text')  
            
            title = title_tag.span.text.strip() if title_tag else None
            
            if title:
                return title
            else:
                raise ValueError("Title not found in the provided HTML.")
        
        except Exception as e:
            print("Failed to get title:", e)
    """
    def get_first_page_summary(self, soup):
        try:
            plot_span = soup.find("span", {"data-testid": "plot-1"})  # Finding the span element
            print("plot_span:", plot_span)  # Debugging statement
            if plot_span:
                # Check if the parent paragraph is closed, if so, open it
                plot_paragraph = plot_span.find_parent("p", {"data-testid": "plot"})
                print("plot_paragraph:", plot_paragraph)  # Debugging statement
                if plot_paragraph and plot_paragraph.has_attr("closed"):
                    plot_paragraph.attrs.pop("closed")
                return plot_span.text.strip()  # Extracting text content and stripping whitespaces
            else:
                return None  # Return None if span element not found
        except Exception as e:
            print("Failed to get first page small representation:", e)
    """
    def get_director(self, soup):
        try:
            all_items = soup.select('.ipc-metadata-list__item')[0].select('.ipc-metadata-list-item__list-content-item.ipc-metadata-list-item__list-content-item--link')
            final = [str(item.text) for item in all_items]
            return final

        except Exception as e:
            print("Failed to get director:", e)
            
    def get_stars(self, soup):
        try:
            all_items = soup.select('.ipc-metadata-list__item')[2].select('.ipc-metadata-list-item__list-content-item.ipc-metadata-list-item__list-content-item--link')
            final = [str(item.text) for item in all_items]
            return final
        except Exception as e:
            print("failed to get stars")

    def get_writers(self, soup):
        try:
            all_items = soup.select('.ipc-metadata-list__item')[1].select('.ipc-metadata-list-item__list-content-item.ipc-metadata-list-item__list-content-item--link')
            final = [str(item.text) for item in all_items]
            return final

        except Exception as e:
            print("failed to get writers")

    def get_related_links(self, soup):
            
        try:
            more_like_this_section = soup.find('section', {'data-testid': 'MoreLikeThis'})
            movie_links = more_like_this_section.find_all('a', href=True)
            imdb_links = set()

            for link in movie_links:
                if '/title/' in link['href']:
                    full_link = link['href'].split('?')[0]  
                    imdb_links.add(full_link)

            base_url = 'https://www.imdb.com'

            full_urls = [base_url + link for link in imdb_links]
            return full_urls if full_urls else ["No IMDb links found"]
        except Exception as e:
            print("failed to get related links")



    def get_summary(self,url):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        URL: str
            The URL of the webpage

        Returns
        ----------
        List[str]
            The summary of the movie
        """

        URL = self.get_summary_link(url)

        #print(URL)
        


        try:
            response = self.crawl(URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            summaries_section = None
            for section in soup.find_all('section', {'class': 'ipc-page-section'}):
                if section.find('div', {'data-testid': 'sub-section-summaries'}):
                    summaries_section = section
                    break

            if summaries_section:
                li_elements = summaries_section.find_all('li', role='presentation')

                summaries = []
                for li in li_elements:
                    inner_div = li.find('div', class_='ipc-html-content-inner-div')
                    if inner_div:
                        summary = inner_div.text.strip()
                        summaries.append(summary)

                #print(summaries)
                return summaries
            else:
                print("Summaries section not found on the page.")
                return []

        except Exception as e:
            print("Failed to get summaries:", e)
            return []

 #   def get_summary(self, soup,url):
 #       res=requests.get(self.get_summary_link(url),self.headers)
        
 #       try:
 #           summaries = [summary.text.strip() for summary in soup.find_all('li', class_='ipl-zebra-list__item')]
 #           return summaries
 #       except Exception as e:
 #           print("failed to get summary")

    def get_synopsis(self, url):


        URL = self.get_summary_link(url)

        #print(URL)
        


        try:
            response = self.crawl(URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            synopsis_section = None
            for section in soup.find_all('section', {'class': 'ipc-page-section'}):
                if section.find('span', {'id': 'synopsis'}):
                    synopsis_section = section
                    break

            if synopsis_section:
                inner_div = synopsis_section.find('div', class_='ipc-html-content-inner-div')
                if inner_div:
                    parent_tags = [parent.name for parent in inner_div.parents]
                    for tag in reversed(parent_tags):
                        inner_div = f"<{tag}>{inner_div}</{tag}>"

                    synopsis = BeautifulSoup(inner_div, 'html.parser').text.strip()
                    synopsis = synopsis.replace("<[document]>", "")

                    #print([synopsis])
                    return [synopsis]

            print("Synopsis section not found on the page.")
            return []

        except Exception as e:
            print("Failed to get synopsis:", e)
            return []
    def get_reviews_with_scores(self, soup, URL):

        response=self.crawl(self.get_review_link(URL))

        soup = BeautifulSoup(response.content, "html.parser")


        try:
            reviews = []
            c=1
            for item in soup.find_all('div', class_='lister-item-content'):
                if(c==12):
                    break
                c=c+1
                text = item.find('div', class_='text').get_text(strip=True)
                score_element = item.find('span', class_='rating-other-user-rating')
                score = score_element.find('span').text.strip() if score_element else 'Score not found'
                reviews.append([ text,  score])
            return reviews



        #    reviews_with_scores = [[review.text.strip(), review.find('span').text.strip()] for review in soup.find_all('div', class_='text show-more__control')]
        #    return reviews_with_scores
        except Exception as e:
            print("failed to get reviews")








    def get_genres(self, soup):
        try:



            script_tag = soup.find_all('script', type='application/ld+json')
            if script_tag:
                data = json.loads(script_tag[0].string)
                x= data['genre']
                #print(x)
                return x



            #genres_list = soup.find('li', {'data-testid': 'storyline-genres'}).find_all('a', class_='ipc-metadata-list-item__list-content-item')
            
            # Extract the text of the budget element

            #print("WTF", genres_list)
            #genres = genres_list.get_text(strip=True)
            
            # Return the budget
            #print(budget)
            #return genres




#            genres_container = soup.find(attrs={"data-testid": "storyline-genres"})
#            if genres_container:
#                genres_links = genres_container.find_all("a", {"class": "ipc-metadata-list-item__list-content-item--link"})
            
#                genres = [link.text.strip() for link in genres_links]
#                return genres
        except Exception as e:
            print("Failed to get genres:", e)

    def get_rating(self,soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            rating_div = soup.find('div', class_='sc-bde20123-2')
            rating = rating_div.get_text(strip=True)
            return rating
        except Exception as e:
            print("failed to get rating:", e)

    def get_mpaa(self,soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            return soup.findAll(attrs={"class":"ipc-link ipc-link--baseAlt ipc-link--inherit-color"})[-1].text
        except Exception as e:
            print("Failed to get MPAA:", e)
            return None




    def get_release_year(self,soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """






        try:
            return soup.findAll(attrs={"data-testid": "title-details-releasedate"})[0].findAll(attrs={"class":"ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link"})[0].text.split(", ")[1].split()[0]


        except Exception as e:
            print("Failed to get release year:", e)
            return None

    def get_languages(self,soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            language_element = soup.find("li", {"data-testid": "title-details-languages"})
            
            languages = []


            if language_element:
                languages_container = language_element.find("div", {"class": "ipc-metadata-list-item__content-container"})
                if languages_container:
                    languages = languages_container.text.strip().split(",")
            languages=re.findall('[A-Z][^A-Z]*', languages[0])

            return languages



        except Exception as e:
            print("Failed to get languages:", e)
            return None

    def get_countries_of_origin(self,soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            countries_element = soup.find("li", {"data-testid": "title-details-origin"})
            if not countries_element:
                countries_element = soup.find("li", {"data-testid": "title-details-country-origin"})
            countries_of_origin = []
            if countries_element:
                countries_container = countries_element.find("div", {"class": "ipc-metadata-list-item__content-container"})
                if countries_container:
                    countries_of_origin = countries_container.text.strip().split(", ")
            return countries_of_origin
        except Exception as e:
            print("Failed to get countries of origin:", e)
            return None

    def get_budget(self,soup,URL):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            budget_element = soup.find("li", {"data-testid": "title-boxoffice-budget"})
            
            budget = budget_element.get_text(strip=True)
            budget = budget.replace("Budget", "")
            
    
            return budget
        except Exception as e:
            print("Failed to get budget:", e, URL)
    
    def get_gross_worldwide(self,soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters 
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """

        try:
            gross_worldwide_element = soup.find("li", {"data-testid": "title-boxoffice-cumulativeworldwidegross"})
            
            gross_worldwide = gross_worldwide_element.get_text(strip=True)
            gross_worldwide = gross_worldwide.replace("Gross worldwide", "")


            return gross_worldwide
        
        except Exception as e:
            print("Failed to get gross worldwide:", e)
            return None







#        try:
#            box_office_section = soup.find('section', {'id': 'box_office'})
#            gross_worldwide = box_office_section.find(text="Gross worldwide:").find_next_sibling().text.strip()
#            return gross_worldwide
#        except Exception as e:
#            print("Failed to get gross worldwide:", e)
#            return None







def main():
    imdb_crawler = IMDbCrawler()
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
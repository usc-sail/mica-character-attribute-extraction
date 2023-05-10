"""Scrape movie scripts from scripts on screen website"""

from narrative_understanding.scrapper_scripts_on_screen.cache import GetPage

import pandas as pd
import requests
import os
import re
import datetime
import pdftotext
from absl import app
from absl import flags
import tqdm


# cache_dir is the directory where successfully retrieved webpages are cached
# movie_scripts_dir is the directory where movie scripts (pdfs and texts) are saved, along with their index
cache_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/movie_scripts/scrape")
movie_scripts_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/movie_scripts/movie_scripts")

# define command-line flags
# --cache to specify a different cache_dir
# --movie_scripts_dir to specify a different movie_scripts_dir
FLAGS = flags.FLAGS
flags.DEFINE_string("cache", default=cache_dir, help="Cache for scraped webpages.")
flags.DEFINE_string("movie_scripts_dir", default=movie_scripts_dir, help="Movie Scripts directory.")
flags.DEFINE_bool("retry", default=False, help="Retry url even if past requests to the url has failed.")
flags.DEFINE_bool("check_for_updates", default=False, help="Check for new script updates on scripts-on-screen website "
                  "(Requests the scripts-on-screen index and movie category pages again even if they are cached)")
flags.DEFINE_integer("timeout", default=60, help="Timeout in seconds for http(s) requests")


def write_text(text: str, text_file: str):
    """Write `text` to the `text_file` text file."""
    with open(text_file, "w") as fw:
        fw.write(text)


def download_pdf(pdf_url: str, pdf_file: str, timeout: int) -> bool:
    """Download pdf from the `pdf_url` url (must end in .pdf or .PDF), save the document to the `pdf_file` path, 
    convert the pdf file to a text file, and save the text file to the same directory as the `pdf_file` path 
    with the same filename but with the .txt extension.

    Args:
        pdf_url (str) : url to a pdf document.
        pdf_file (str) : filepath where pdf document will be saved.
        timeout (int) : timeout in seconds
    
    Returns:
        success flag (bool) : True if the pdf is successfully downloaded and converted to a text file, else False.
    """
    pdf_file = pdf_file if re.search(r"\.pdf$", pdf_file) is not None else pdf_file + ".pdf"
    text_file = re.sub(r"\.pdf$", ".txt", pdf_file)
    try:
        response = requests.get(pdf_url, timeout=timeout)
        if response.status_code == 200:
            with open(pdf_file, "wb") as fw:
                fw.write(response.content)
            with open(pdf_file, "rb") as fr:
                pdf = pdftotext.PDF(fr)
            text = "\n\n".join(pdf)
            n_words = len(text.split())
            assert n_words > 1000
            write_text(text, text_file)
            return True
    except Exception:
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
        if os.path.exists(text_file):
            os.remove(text_file)
    return False


def get_movie_script(url: str, getpage: GetPage, retry: bool) -> str | None:
    """If `url` links to a PDF script, return it. Else find the script (PDF url or text) within the `url` webpage.
    
    Args:
        url (str) : url 
        getpage (GetPage) : GetPage object to retrieve webpages
        retry (bool) : retry requesting the url even if past requests have failed (passed to getpage calls)

    Returns:
        a pdf url or script text, None if some error occurs.
    """
    # url links to a PDF document
    if url.endswith("pdf") or url.endswith("PDF"):
        return url

    # retrieve page
    soup = getpage(url, retry_error_url=retry)

    # script text
    text = ""

    if soup is not None:
        # TEXT document
        if url.endswith("txt") or url.endswith("TXT"):
            text = soup.text

        # scriptslug
        elif re.search("scriptslug.com", url) is not None:
            for a_element in soup.find_all("a"):
                if a_element.text.lower().strip() == "read the script":
                    pdf_url = a_element["href"]
                    if pdf_url.endswith("pdf") or pdf_url.endswith("PDF"):
                        return pdf_url
        
        # imsdb html script, dailyscript html script, screenplays for you
        elif re.search(r"((imsdb)|(dailyscript)|(horrorlair))\.com.*\.html?$", url) or (
                re.search("sfy.ru", url) is not None):
            preformatted_element = soup.find("pre")
            if preformatted_element is not None:
                text = preformatted_element.text
        
        # dailyscript html containing text without <pre> tag
        elif re.search(r"dailyscript.*\.html?", url):
            text = soup.text
    
    if len(text.split()) > 1000:
        return text

    return None

def scrape_scripts_on_screen(_):
    # scripts on screen url, current date, and command-line flags
    scripts_on_screen_url = "https://scripts-onscreen.com/movie-script-index/"
    date_accessed = datetime.datetime.now().strftime("%b %d %Y")
    cache_dir = FLAGS.cache
    movie_scripts_dir = FLAGS.movie_scripts_dir
    timeout = FLAGS.timeout
    retry = FLAGS.retry
    update = FLAGS.check_for_updates

    # initialize GetPage object to retrieve, parse, and cache webpages
    getpage = GetPage(cache_dir, timeout=timeout)

    # get scripts on screen index page and find the category pages of movie links
    index_soup = getpage(scripts_on_screen_url, retry_error_url=retry)
    index_links_soup = index_soup.find("div", attrs={"class":"soslinks"})
    links = [a_element["href"] for a_element in index_links_soup.find_all("a")]
    print("parsed Scripts On Screen index page\n")

    # get the category pages and find the movie links
    link_soups = getpage(*links, retry_error_url=retry)
    movie_links = []
    for link_soup in link_soups.values():
        list_elements = link_soup.find("div", {"class": "sosindex"}).find_all("li")
        for list_element in list_elements:
            movie_link = "https://scripts-onscreen.com" + list_element.find("a")["href"]
            movie_links.append(movie_link)
    print("retrieved Scripts on Screen movie link category pages\n")

    # get the movie pages
    movie_pages = getpage(*movie_links, retry_error_url=retry, update_cache=update)
    print(f"retrieved {len(movie_pages)} movie pages\n")

    # populate the movie scripts index with existing entries
    # the index is a mapping from script url to filename (without the extension), download date, IMDB id, MOVIEDB id,
    # and the synopsis on scripts on screen website
    movie_scripts_index_file = os.path.join(movie_scripts_dir, "index.csv")
    movie_scripts_index = {}
    next_file = 0
    if os.path.exists(movie_scripts_index_file):
        movie_index_df = pd.read_csv(movie_scripts_index_file, index_col=None)
        for _, row in movie_index_df.iterrows():
            movie_scripts_index[row["url"]] = (row["file"], row["date"], row["imdb_id"], row["moviedb_id"], 
                                               row["script_on_screen_synopsis"])
            next_file = max(next_file, row["file"] + 1)

    # error urls
    error_urls = set()
    error_urls_file = os.path.join(movie_scripts_dir, "error_urls.txt")
    if os.path.exists(error_urls_file):
        with open(error_urls_file, "r") as fr:
            for line in fr:
                error_urls.add(line)
    
    # counter for correct urls
    n_correct_urls = 0

    # loop over each movie page
    tbar = tqdm.tqdm(enumerate(movie_pages.values()), total=len(movie_pages))
    for i, movie_page in tbar:
        prefix = (f"movie {i + 1:5d}/{len(movie_pages)}: {n_correct_urls:5d} correct urls, "
                  f"{len(error_urls):5d} error urls")
        tbar.set_description(prefix)
        main_div_soup = movie_page.find("div", {"class": "main_div"})
        synopsis, imdb_id, moviedb_id = None, None, None

        # get scripts on screen synopsis
        movie_synopsis_soups = [movie_prop_soup 
                                for movie_prop_soup in main_div_soup.find_all("div", {"class": "movie-prop"}) 
                                    if movie_prop_soup.text.startswith("Script Synopsis")]
        if movie_synopsis_soups:
            synopsis = re.sub(f"^Script Synopsis:", "", movie_synopsis_soups[0].text).strip()

        # get script urls
        movie_links_soup = main_div_soup.find("div", {"class": "movie-links"})
        urls = []
        if movie_links_soup is not None:
            list_elements = movie_links_soup.find_all("li")

            # loop over each link
            # find imdb and moviedb id
            # find all script links that are not paid and does not contain the word 'Transcript' in the text
            for list_element in list_elements:
                url = list_element.find("a")["href"]
                text = list_element.text
                imdb_match = re.search(r"\(\s*(tt\d+)\s*\)", text)
                moviedb_match = re.search(r"\(\s*(\d+)\s*\)", text)
                if "IMDb" in text and imdb_match is not None:
                    imdb_id = imdb_match.group(1)
                elif "TheMovieDB.org" in text and moviedb_match is not None:
                    moviedb_id = moviedb_match.group(1)
                elif "$" not in text and "Transcript" not in text:
                    urls.append(url)

        # get pdf urls or scrape the texts
        # script url is the url on the movie page of scripts on screen used as index
        # script url might not always be the same as the pdf url, for cases where more webpages needs to be retrieved
        # to find the actual url to the movie script
        script_url_to_pdf_url, script_url_to_text = {}, {}
        for url in urls:
            tbar.set_description(f"{prefix}: GET {url:100s}")
            if url in movie_scripts_index:
                n_correct_urls += 1
            elif retry or url not in error_urls:
                response = get_movie_script(url, getpage, retry=retry)
                if response is not None:
                    if re.match(r"http", response) is not None:
                        script_url_to_pdf_url[url] = response
                    else:
                        script_url_to_text[url] = response
                else:
                    error_urls.add(url)

        # download pdfs
        for script_url, pdf_url in script_url_to_pdf_url.items():
            pdf_file = os.path.join(movie_scripts_dir, f"{next_file}.pdf")
            tbar.set_description(f"{prefix}: DOWN {pdf_url:100s}")
            if download_pdf(pdf_url, pdf_file, timeout=timeout):
                movie_scripts_index[script_url] = (next_file, date_accessed, imdb_id, moviedb_id, synopsis)
                n_correct_urls += 1
                next_file += 1
            else:
                error_urls.add(script_url)

        # write texts
        for script_url, text in script_url_to_text.items():
            movie_scripts_index[script_url] = (next_file, date_accessed, imdb_id, moviedb_id, synopsis)
            text_file = os.path.join(movie_scripts_dir, f"{next_file}.txt")
            n_correct_urls += 1
            next_file += 1
            write_text(text, text_file)

        # write movie scripts index and cache index periodically
        if (i + 1) % 10 == 0 or i == len(movie_pages) - 1:
            rows = []
            for url, (file_id, date, imdb_id, moviedb_id, synopsis) in movie_scripts_index.items():
                row = [url, file_id, date, imdb_id, moviedb_id, synopsis]
                rows.append(row)
            movie_index_df = pd.DataFrame(rows, columns=["url", "file", "date", "imdb_id", "moviedb_id", 
                                                         "script_on_screen_synopsis"])
            movie_index_df.to_csv(movie_scripts_index_file, index=False)
            getpage.write_index()
            with open(error_urls_file, "w") as fw:
                fw.write("\n".join(error_urls))


if __name__ == '__main__':
    app.run(scrape_scripts_on_screen)
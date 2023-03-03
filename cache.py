"""Retrieve webpage, parse it using beautiful soup, cache the webpage, and return the parsed soup."""
import requests
import pandas as pd
import os
import datetime
import bs4
import tqdm

class GetPage(object):
    """Retrieve webpage, parse it using beautiful soup, cache the webpage, and return the parsed beautiful soup
    object"""

    def __init__(self, cache_dir: str = None, timeout: int = 60) -> None:
        """Initializer for GetPage instances.

        Params:
            cache_dir (str) : Directory where scraped webpages are cached. If not provided, webpages are not cached.
            timeout (int) : Timeout in seconds while requesting a webpage.
        """
        self._cache_dir: str = cache_dir
        self._timeout = timeout
        if self._cache_dir is not None:
            # self._cache is mapping from url (str) to tuple of filename (int) and date accessed (str).
            # self._error_urls is the set of urls that getpage tried requesting in the part but ended in an error.
            # self._next_file is the integer filename of the next webpage that will be cached.
            # self._date_accessed is the current date in string format. This is the date assigned to new cached
            # webpages.
            # self._cache_index_file is the filepath of the cache index, a dataframe containing columns for
            # url (str), filename (integer), and date accessed (str).
            # self._error_urls_file is the filepath of the text file that includes the past error urls.
            self._cache: dict[str, tuple[int, str]] = {}
            self._error_urls: set[str] = set()
            self._next_file: int = 0
            self._date_accessed: str = datetime.datetime.now().strftime("%b %d %Y")
            self._cache_index_file = os.path.join(self._cache_dir, "index.csv")
            self._error_urls_file = os.path.join(self._cache_dir, "error_urls.txt")

            # read the cache
            if os.path.exists(self._cache_index_file):
                cache_index_df = pd.read_csv(self._cache_index_file, index_col=None)
                for _, row in cache_index_df.iterrows():
                    self._cache[row["url"]] = (row["file"], row["date"])
                if len(cache_index_df):
                    self._next_file = cache_index_df["file"].max() + 1
            
            # read the past error urls
            if os.path.exists(self._error_urls_file):
                with open(self._error_urls_file, "r") as fr:
                    for line in fr:
                        self._error_urls.add(line.strip())

    def __call__(self, *urls, retry_error_url=False, disable_progress_bar=False) -> (
            dict[str, bs4.BeautifulSoup] | bs4.BeautifulSoup | None):
        """Retrieve webpages for the given urls and parse them using beautifulsoup.

        Params:
            urls : Webpage urls.
            retry_error_url : Retry requesting the url even if it was already requested in the past which ended in an
                error.
            disable_progress_bar : Disable progress bar. Automatically disabled for a single url.
        
        Returns:
            A dictionary of webpage url to beautifulsoup parsed object. If url could not be requested, the url is 
            absent from the dictionary. If a single url is given, a beautifulsoup object or None is returned.
        """
        pages: dict[str, bs4.BeautifulSoup] = {}
        urlbar = tqdm.tqdm(urls, unit="url") if not disable_progress_bar and len(urls) > 1 else urls
        for url in urlbar:
            if not isinstance(urlbar, tuple):
                urlbar.set_description(f"{url:100s}")
            if self._cache_dir is not None and url in self._cache:
                    with open(os.path.join(self._cache_dir, str(self._cache[url][0])), "rb") as fr:
                        pages[url] = bs4.BeautifulSoup(fr.read(), "html.parser")
            elif self._cache_dir is None or retry_error_url or url not in self._error_urls:
                try:
                    response = requests.get(url, "html.parser", timeout=self._timeout)
                    if response.status_code == 200:
                        pages[url] = bs4.BeautifulSoup(response.content, "html.parser")
                        if self._cache_dir is not None:
                            with open(os.path.join(self._cache_dir, str(self._next_file)), "wb") as fw:
                                fw.write(response.content)
                            self._cache[url] = (self._next_file, self._date_accessed)
                            self._next_file += 1
                    else:
                        if self._cache_dir is not None:
                            self._error_urls.add(url)
                except Exception:
                    self._error_urls.add(url)
        if len(urls) > 10:
            self.write_index()
        if len(urls) == 1:
            if pages:
                return pages[urls[0]]
            else:
                return None
        return pages

    def write_index(self):
        """Write cache index and error urls. The cache index maps urls to filenames and date accessed."""
        if self._cache_dir is not None:
            rows = []
            for url, (file, date) in self._cache.items():
                rows.append([file, url, date])
            df = pd.DataFrame(rows, columns=["file", "url", "date"])
            df.to_csv(self._cache_index_file, index=False)
            with open(self._error_urls_file, "w") as fw:
                fw.write("\n".join(sorted(self._error_urls)))

    def __del__(self):
        """Write cache index before destroying object"""
        self.write_index()
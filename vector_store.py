import bs4
import urllib.request
from urllib.request import Request, urlopen
from langchain_utils import split_doc, create_redis_index
from langchain.embeddings import OpenAIEmbeddings

def build_vectorstore(link):
    # retrieve_web_content(link)
    docs = split_doc(path="./web_data/", path_type="dir")
    # link = 'https://www.themuse.com/advice/43-resume-tips-that-will-help-you-get-hired'
    rds = create_redis_index(docs, OpenAIEmbeddings(), link, "redis_cl_test")
    # add_redis_index(docs, OpenAIEmbeddings, link, "redis_index")
    print(rds)
    

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

def retrieve_web_content(link):

    req = Request(
        url=link, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    try: 
        webpage=str(urllib.request.urlopen(link).read())
    except Exception: 
        # webpage = urlopen(req).read()
        opener = AppURLopener()
        webpage = opener.open(link)
    soup = bs4.BeautifulSoup(webpage, features="lxml")

    content = soup.get_text()


    if content:
        with open('./web_data/cl_indeed.txt', 'w') as file:
            file.write(content)
            file.close()
            print('Content retrieved and written to file.')
    else:
        print('Failed to retrieve content from the URL.')


retrieve_web_content("https://www.thebalancemoney.com/things-not-to-include-in-cover-letter-2060284")

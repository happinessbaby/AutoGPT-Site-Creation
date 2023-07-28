import bs4
import urllib.request
from urllib.request import Request, urlopen
from langchain_utils import split_doc, create_redis_index, create_redis_index_with_source
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


resume_sample_path = "./resume_samples/"
resume_advice_path = "./web_data/resume/"
resume_sample_index = "redis_resume_sample"
resume_advice_index = "redis_resume_advice"
cover_letter_advice_path = "./web_data/cover_letter/"
cover_letter_advice_index = "redis_cover_letter_advice"


def create_vector_store(index_name, path, path_type="file", source=None):
    # retrieve_web_content(link)
    docs = split_doc(path=path, path_type=path_type)
    # link = 'https://www.themuse.com/advice/43-resume-tips-that-will-help-you-get-hired'
    if (source!=None):
        rds = create_redis_index_with_source(docs, OpenAIEmbeddings(), source, index_name)
    else:
        rds = create_redis_index(docs, OpenAIEmbeddings(), index_name)
    # add_redis_index(docs, OpenAIEmbeddings, link, "redis_index")
    print(rds)

    



if __name__ == '__main__':
    create_vector_store(resume_advice_index, resume_advice_path, path_type="dir" )

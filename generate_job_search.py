import os
import markdown
from openai_api import get_completion
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

chat = ChatOpenAI(temperature=0.0)
embeddings = OpenAIEmbeddings()

def find_similar_jobs(job_title):

    loader = CSVLoader(file_path="jobs.csv")
    docs = loader.load()

    db = DocArrayInMemorySearch.from_documents(
        docs, 
        embeddings
    )
    # doing Q&A with llm
    retriever = db.as_retriever()

    query = f"""List all the jobs related to or the same as {job_title} in a markdown table.
    
    Do not make up things not in the given context. """

    # docs = db.similarity_search(query)

    # print(len(docs))
    # print(docs[0])

    qa_stuff = RetrievalQA.from_chain_type(
    llm=chat, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
    )

    response = qa_stuff.run(query)
    print(response)
    return response


#TBD: based on resume information, suggest jobs 
def generate_job_suggestion():
    return None

my_job_title = 'software developer'
if __name__ == '__main__':
    find_similar_jobs(my_job_title)
from langchain.agents.react.base import DocstoreExplorer
from langchain.document_loaders import TextLoader
from langchain.docstore.wikipedia import Wikipedia
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.agents import Tool
from langchain.tools.python.tool import PythonREPLTool
from langchain.utilities import SerpAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper
from langchain import ElasticVectorSearch
from langchain.vectorstores.elastic_vector_search import ElasticKnnSearch
from langchain.embeddings import ElasticsearchEmbeddings
from elasticsearch import Elasticsearch
from ssl import create_default_context
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Union
import re
from langchain import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import subprocess
import sys
from feast import FeatureStore


# You may need to update the path depending on where you stored it
feast_repo_path = "."

def get_index(file):
    loader = TextLoader(file, encoding='utf8')
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])
    return index


def create_wiki_tools():
    docstore = DocstoreExplorer(Wikipedia())
    tools = [
        Tool(
            name = "Search",
            func = docstore.search,
            description= "Search for a term in the docstore."
        ),
        Tool(
            name = "Lookup",
            func = docstore.lookup,
            description = "Lookup a term in the docstore."
        ),
    ]
    return tools

def create_qa_tools(qa_chain):
    # _index = get_index(document)
    tools = [
        Tool(
            name="QA Document",
            # func=lambda q: str(_index.query(q)),
            func = qa_chain.run,
            description="Useful for answering questions found in documents",
            return_direct=True,
        ),
    ]
    return tools

def create_doc_tools(doc):
    index = get_index(doc)
    tools = [
        Tool(
        name = f"{doc} Document",
        func = lambda q: str(index.query(q)),
        description="Useful for giving personalized answers",
        return_direct=True,
        )
    ]
    return tools


def create_search_tools(name, top_n):
    if (name=="google"):
        search = GoogleSearchAPIWrapper(k=top_n)
        tool = [
            Tool(
            name = "Google Search", 
            description= "useful for when you need to ask with search",
            func=search.run,
        ),
        ]
    elif (name=="serp"):
        search = SerpAPIWrapper() 
        tool = [
            Tool(
            name="SerpSearch",
            description= "useful for when you need to ask with search",
            func=search.run,
        ),
        ]
    return tool

def create_db_tools(db_chain, name):
    tool = [
        Tool(
        name="PineCone DB",
        func=db_chain.run,
        description="useful for when you need to answer questions about PineCone DB. Input should be in the form of a question containing full context",
    ),
    ]
    return tool

# def create_process_tools(file_name):
#     tool = [
#         Tool(
#         name = "Python Process",
#         func = subprocess.run([f"{sys.executable}", f"{file_name}"]),
#         description="useful for when you are asked to write a cover letter",
#         ),
#     ]
#     return tool



def create_QA_chain(chat, embeddings, db_type, docs=None, chain_type="stuff"):
    if (db_type=="chroma"):
        persist_directory = 'myvectordb'
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function = embeddings)
    elif (db_type=="inmemory"):
        vectorstore = DocArrayInMemorySearch.from_documents(
        docs, 
        embeddings
    )
    elif (db_type == "feast"):
        vectorstore = FeatureStore(repo_path=feast_repo_path)

        
    prompt_template = """If the context is not relevant, 
    please answer the question by using your own knowledge about the topic
    
    {context}
    
    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=chat, 
        chain_type=chain_type, 
        retriever=vectorstore.as_retriever(), 
        verbose=True, 
        chain_type_kwargs=chain_type_kwargs,
    )

    return qa


def create_elastic_knn():
    # Define the model ID
    model_id = "mymodel"
  # Create Elasticsearch connection
    context = create_default_context(cafile="/home/tebblespc/Downloads/certs.pem")
    es_connection = Elasticsearch(
    hosts=["https://127.0.0.1:9200"], basic_auth=("elastic", "changeme"), ssl_context = context)   

 # Instantiate ElasticsearchEmbeddings using es_connection
    embeddings = ElasticsearchEmbeddings.from_es_connection(
        model_id,
        es_connection,
    )

    query = "Hello"
    knn_result = knn_search.knn_search(query=query, model_id="mymodel", k=2)
    print(f"kNN search results for query '{query}': {knn_result}")
    print(
        f"The 'text' field value from the top hit is: '{knn_result['hits']['hits'][0]['_source']['text']}'"
    )


    # Initialize ElasticKnnSearch
    knn_search = ElasticKnnSearch(
        es_connection=es_connection, index_name="elastic-index", embedding=embeddings
    )
    
    return knn_search


def create_python_agent(llm):
    agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
    )
    return agent



def get_summary(llm, docs):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary



def add_embedding(embeddings, text):
    query_embedding = embeddings.embed_query(text)
    return query_embedding



# # Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    

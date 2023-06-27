from langchain.agents.react.base import DocstoreExplorer
from langchain.document_loaders import TextLoader
from langchain.docstore.wikipedia import Wikipedia
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.agents import Tool
from langchain.tools.python.tool import PythonREPLTool
from langchain.utilities import SerpAPIWrapper


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

def create_document_tools(document):
    _index = get_index(document)
    tools = [
        Tool(
            name=f"{_index} index",
            func=lambda q: str(_index.query(q)),
            description="Useful to answering questions about the given file",
            return_direct=True,
        ),
    ]
    return tools

def create_search_tools():
    search = SerpAPIWrapper()  
    tool = [
        Tool(
        name="SerpSearch",
        # description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        description= "Useful when you cannot find answers in the docstore",
        func=search.run,
    ),
    ]
    return tool


def create_python_agent(llm):
    agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
    )
    return agent



def add_embedding(embeddings):
    embed = embeddings.embed_query("Prompt Engineer")
    return embed

    
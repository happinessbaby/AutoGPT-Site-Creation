from langchain.agents.react.base import DocstoreExplorer
from langchain.document_loaders import TextLoader
from langchain.docstore.wikipedia import Wikipedia
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.agents import Tool


cover_letter_reference = "file of references for writing good cover letters"

def get_index(file):
    loader = TextLoader(file, encoding='utf8')
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])
    return index


def create_tools():
    docstore = DocstoreExplorer(Wikipedia())
    # cover_letter_index = get_index(cover_letter_reference)
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
        # Tool(
        #     name=f"cover letter index",
        #     func=lambda q: str(cover_letter_index.query(q)),
        #     description="Useful to answering questions about the given file",
        #     return_direct=True,
        # ),
    ]
    return tools

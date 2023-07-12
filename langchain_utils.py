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

def create_google_search_tools(top_n):
    # search = SerpAPIWrapper()  
    search = GoogleSearchAPIWrapper(k=top_n)
    tool = [
        Tool(
        # name="SerpSearch",
        name = "Google Search", 
        # description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        # description= "Useful when you cannot find answers in the docstore or other provided documents",
        description= "useful for when you need to ask with search",
        func=search.run,
    ),
    ]
    return tool

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

def create_custom_llm_agent(llm, tools):
    # Set up the base template
    template = """Complete the objective as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question. Your output should be detailed, descriptive, and at least 100 words. Do not provide just a summary. 


    Begin!

    Question: {input}
    {agent_scratchpad}"""


    prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
    )
    output_parser = CustomOutputParser()
    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    return agent_executor

def get_summary(llm, docs):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)



def add_embedding(embeddings, text):
    query_embedding = embeddings.embed_query(text)
    return query_embedding



# Set up a prompt template
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
    

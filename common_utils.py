from basic_utils import read_txt
from openai_api import get_completion
from openai_api import get_completion, get_completion_from_messages
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.agents import load_tools, initialize_agent, Tool, AgentExecutor
from langchain.document_loaders import CSVLoader, TextLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents import AgentType
from langchain.chains import RetrievalQA,  HypotheticalDocumentEmbedder, LLMChain
from pathlib import Path
from basic_utils import read_txt
from langchain_utils import create_wiki_tools, create_search_tools, create_db_tools, create_compression_retriever, split_doc
from langchain import PromptTemplate
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents.agent_toolkits import create_python_agent
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.tools.convert_to_openai import  format_tool_to_openai_function
from langchain.schema import HumanMessage
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain, StuffDocumentsChain
from langchain.docstore import InMemoryDocstore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_transformers import (
    LongContextReorder,
)
from typing import Any, List
from langchain.docstore.document import Document
from langchain.tools import tool
import os
import sys
import re
import string
import random
import json
import faiss



delimiter = "####"
delimiter2 = "'''"
delimiter3 = '---'
delimiter4 = '////'
categories = ["resume", "cover letter", "job posting", "resume evaluation"]

def extract_personal_information(resume: str,  llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", cache=False)) -> Any:

    name_schema = ResponseSchema(name="name",
                             description="Extract the full name of the applicant. If this information is not found, output -1")
    email_schema = ResponseSchema(name="email",
                                        description="Extract the email address of the applicant. If this information is not found, output -1")
    phone_schema = ResponseSchema(name="phone",
                                        description="Extract the phone number of the applicant. If this information is not found, output -1")
    address_schema = ResponseSchema(name="address",
                                        description="Extract the home address of the applicant. If this information is not found, output -1")


    response_schemas = [name_schema, 
                        email_schema,
                        phone_schema, 
                        address_schema, 
                        ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    template_string = """For the following text, delimited with {delimiter} chracters, extract the following information:

    name: Extract the full name of the applicant. If this information is not found, output -1\

    email: Extract the email address of the applicant. If this information is not found, output -1\

    phone number: Extract the phone number of the applicant. If this information is not found, output -1\
    
    address: Extract the home address of the applicant. If this information is not found, output -1\

    text: {delimiter}{text}{delimiter}

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=template_string)
    messages = prompt.format_messages(text=resume, 
                                format_instructions=format_instructions,
                                delimiter=delimiter)

    
    response = llm(messages)
    personal_info_dict = output_parser.parse(response.content)
    print(f"Successfully extracted personal info: {personal_info_dict}")
    return personal_info_dict



def extract_posting_information(posting: str, llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", cache=False)) -> Any:

    job_schema = ResponseSchema(name="job",
                             description="Extract the job position of the job listing. If this information is not found, output -1")
    company_schema = ResponseSchema(name="company",
                                        description="Extract the company name of the job listing. If this information is not found, output -1")
    
    response_schemas = [job_schema, 
                        company_schema]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    template_string = """For the following text, delimited with {delimiter} chracters, extract the following information:

    job: Extract the job positiong of the job posting. If this information is not found, output -1\

    company: Extract the company name of the job posting. If this information is not found, output -1\


    text: {delimiter}{text}{delimiter}

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=template_string)
    messages = prompt.format_messages(text=posting, 
                                format_instructions=format_instructions,
                                delimiter=delimiter)
 
    response = llm(messages)
    posting_info_dict = output_parser.parse(response.content)
    print(f"Successfully extracted posting info: {posting_info_dict}")
    return posting_info_dict



def extract_job_title(resume: str) -> str:
    response = get_completion(f"""Read the resume closely. It is delimited with {delimiter} characters. 
                              
                              Output a likely job position that this applicant is currently holding or a possible job position he or she is applying for.
                              
                              resume: {delimiter}{resume}{delimiter}. \n
                              
                              Response with only the job position, no punctuation or other text. """)
    print(f"Successfull extracted job title: {response}")
    return response



def extract_fields(resume: str, llm=OpenAI(temperature=0, cache=False)) -> str:

    query =  """Search and extract fields of the resume delimited with {delimiter} characters.

         Some common resume fields include but not limited to personal information, objective, education, work experience, awards and honors, and skills.
         
         resume: {delimiter}{resume}{delimiter} \n
         
         {format_instructions}"""
    
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template=query,
        input_variables=["delimiter", "resume"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    _input = prompt.format(delimiter=delimiter, resume = resume)
    response = llm(_input)
    print(f"Successfully extracted fields: {response}")
    return response

def get_field_content(resume: str, field: str) -> str:
     
   query = f"""Retrieve all the content of field {field} from the resume file delimiter with {delimiter} charactres.

      resume: {delimiter}{resume}{delimiter}
    """
   response = get_completion(query)
   print(f"Successfully got field content: {response}")
   return response

##TODO: This is currently highly wild
def expand_qa(read_path: str) -> str:

    agent_executor = create_python_agent(
          llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
          tool=PythonREPLTool(),
          verbose=True,
          agent_type=AgentType.OPENAI_FUNCTIONS,
          agent_executor_kwargs={"handle_parsing_errors": True},
      )
    content = read_txt(read_path)
    missing_stuff=agent_executor.run(f"""
                                    Find distinct texts in brackets. Do not output anything that's not bracketed:
                                    {content}""")
    
    system_message = f""" You are an assistant that generate questions for missing information in the content.

        The missing things are delimited by {delimiter} characters.

        missing things: {delimiter}{missing_stuff}{delimiter}

        For each missing thing, generate a question to ask the user given the content context. 
        """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': content}
    ]
    response = get_completion_from_messages(messages)
    print(f"Sucessfully expanded questions: {response}")
    return response


    
def search_related_samples(job_title: str, directory: str) -> List[str]:

    system_message = f"""
		You are an assistant that evaluates whether the job position described in the content is similar to {job_title} or relevant to {job_title}. 

		Respond with a Y or N character, with no punctuation:
		Y - if the job position is similar to {job_title} or relevant to it
		N - otherwise

		Output a single letter only.
		"""
    related_files = []
    for path in  Path(directory).glob('**/*.txt'):
        file = str(path)
        content = read_txt(file)
        # print(file, len(content))
        messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': content}
        ]	
        
        response = get_completion_from_messages(messages, max_tokens=1)
        if (response=="Y"):
            related_files.append(file)
    if len(related_files)==0:
        related_files.append(directory + "software_developer.txt")
    return related_files




def get_web_resources(query: str, llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-0613", cache=False)) -> str:

    search = GoogleSearchAPIWrapper()
    embedding_size = 1536  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore = FAISS(OpenAIEmbeddings().embed_query, index, InMemoryDocstore({}), {})
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm, 
        search=search, 
    )
    # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=web_research_retriever)
    # result = qa_chain({"question": query})
    # print(f"successfully got web resources: {result}")
    # return result.get("answer")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=web_research_retriever)
    response = qa_chain.run(query)
    print(f"Successfully got web resources: {response}")
    return response


    
def reorder_docs(retriever: Any, query: str) -> List[Document]:
    reordering = LongContextReorder()
    compressed_docs = retriever.get_relevant_documents(query)
    reordered_docs = reordering.transform_documents(compressed_docs)
    return reordered_docs

@tool
def create_n_docs_tool(query: str) -> str:
    """Searches for relevant documents that may contain the answer to the query."""
    subquery_relevancy = "relevancy in resume"
    compression_retriever = create_compression_retriever()
    # tool_description = """This is useful when defining relevancy, use only needed. 
    # If input has enough information to determine relevancy, you don't need to use this tool. """
    reordered_docs = reorder_docs(compression_retriever, subquery_relevancy)
    texts = [doc.page_content for doc in reordered_docs]
    texts_merged = "\n\n".join(texts)
    return texts_merged

def retrieve_from_db(query: str, llm=OpenAI(temperature=0.8, cache=False)) -> str:

    retriever = create_compression_retriever()
    reordered_docs = reorder_docs(retriever, query)

    # Option 2: 
    # chain = RetrievalQA.from_chain_type(
    #     llm, 
    #     chain_type="map_reduce", 
    #     # verbose=True, 
    #     retriever=retriever,
    #     # input_key="question"
    #     )
    # response = chain.run(query)
    # Option 3:
    # We prepare and run a custom Stuff chain with reordered docs as context.
    # Override prompts
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    document_variable_name = "context"
    stuff_prompt_override = """Given this text extracts:
    -----
    {context}
    -----
    Please answer the following question:
    {query}"""
    prompt = PromptTemplate(
        template=stuff_prompt_override, input_variables=["context", "query"]
    )

    # Instantiate the chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )
    response = chain.run(input_documents=reordered_docs, query=query, verbose=True)

    print(f"Successfully retrieved advice: {response}")
    return response

    
def get_summary(posting_path: str, prompt_template = "", llm=OpenAI()) -> str:
    docs = split_doc(path=posting_path, path_type="file")
    prompt_template = """Identity the job position, company then provide a summary of the following job posting:
        {text} \n
        Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
      """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    response = chain.run(docs)
    print(f"Sucessfully got summary: {response}")
    return response


def create_sample_tools(related_samples: List[str], sample_type: str, llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", cache=False)) -> str:

    tools = []
    for file in related_samples:
        # initialize the bm25 retriever and faiss retriever for hybrid search
        # https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble
        docs = split_doc(file, "file")
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 2
        faiss_retriever = FAISS.from_documents(docs, OpenAIEmbeddings()).as_retriever(search_kwargs={"k": 2})
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        # name and description of the tool are really important in the agent using the tool
        tool_description = f"This is a {sample_type} sample. Use it to compare with other {sample_type} samples"
        tool = create_db_tools(llm, ensemble_retriever, f"{sample_type}_{random.choice(string.ascii_letters)}", tool_description)
        tools.extend(tool)
    print(f"Successfully created {sample_type} tools")
    return tools


def generate_multifunction_response(query: str, tools: List[Tool], llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", cache=False)) -> str:

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_MULTI_FUNCTIONS, 
        max_iterations=2,
        early_stopping_method="generate",
        # verbose=True
    )
    response = agent({"input": query}).get("output", "")   
    print(f"Successfully got multifunction response: {response}")
    return response


def categorize_content(content):

    system_message = f"""
        You are an assistant that categorizes content of text based on a list of categories. 
            
        There may be other irrelevant content in the text. Ignore them and ignore all formatting. 
            
        The provided categories are : {str(categories)}

        Respond with the category name, if the content contains text that belongs to a provided category, with no punctuation:

        Output the category name only. If the content does not belong to any of the provided categories, output -1.

        """

    messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': content}
    ]	

    response = get_completion_from_messages(messages, max_tokens=10)

    return response
	







    
















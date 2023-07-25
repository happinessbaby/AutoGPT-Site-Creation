from basic_utils import read_txt
from openai_api import get_completion
from openai_api import get_completion, evaluate_response
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.agents import load_tools, initialize_agent, Tool, AgentExecutor
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents import AgentType
from langchain.chains import RetrievalQA
from pathlib import Path
from basic_utils import check_content_safety, read_txt, retrieve_web_content
from langchain_utils import create_wiki_tools, create_search_tools, create_QA_chain, create_qa_tools, split_doc, create_redis_index, add_redis_index
from langchain import PromptTemplate
import sys



delimiter = "####"
delimiter2 = "'''"
delimiter3 = '---'
delimiter4 = '////'

def extract_personal_information(llm, resume):

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
                        address_schema]

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
    print(personal_info_dict.get('email'))
    return personal_info_dict


def fetch_samples(llm, embeddings, job_title, samples):

    table = find_similar_jobs(llm, embeddings, job_title)

    prompt = f"""Extract the Job Title values in the markdown table: {table}.
    
    Output your answer as a comma separated list. If there is no table, return -1. """

    jobs = get_completion(prompt)
    sample_string = ""
    print(jobs)
    if (jobs!=-1):
        jobs_list = jobs.split(",")
        for job in jobs_list:
            if (samples.get(job)!=None):
                sample = read_txt(samples.get(job))
                sample_string = sample_string + "\n" + f" {delimiter3}\n{sample}\n{delimiter3}" + "\n\nexample:"   
    # print(sample_string)
    return sample_string


def find_similar_jobs(llm, embeddings, job_title):

    loader = CSVLoader(file_path="jobs.csv")
    docs = loader.load()

    qa_stuff = create_QA_chain(llm, embeddings, docs=docs, db_type = "docarray")

    query = f"""List all the jobs related to or the same as {job_title} in a markdown table.
    
    Do not make up things not in the given context. """
    

    response = qa_stuff.run(query)
    print(response)
    return response


# okay for basic version, but in the future, need a vecstore where all the information are saved and updated
def get_web_resources(llm, query, top=10):

    # wiki_tools = create_wiki_tools()
    # SERPAPI has limited searches, for now, use plain google
    tools = create_search_tools("google", top)
    # tools = wiki_tools+search_tools
    agent= initialize_agent(
        tools, 
        llm, 
        # agent=AgentType.REACT_DOCSTORE,
        agent="zero-shot-react-description",
        handle_parsing_errors=True,
        verbose = True,
        )
    try:
        response = agent.run(query)
        return response
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            print(e)
            raise e
        response = response.removeprefix(
            "Could not parse LLM output: `").removesuffix("`")
        return response
    

def retrieve_from_vectorstore(llm, embeddings, query):
    qa_stuff = create_QA_chain(llm, embeddings, db_type="redis")
    # Option 1: qa tool + react 
    # Option 1 seems to give better advices 
    tools = create_qa_tools(qa_stuff)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    try:
        response = agent.run(query)
        return response
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            print(e)
            raise e
        response = response.removeprefix(
            "Could not parse LLM output: `").removesuffix("`")
        return response
    response = agent.run(query)
    # Option 2
    # response = qa_stuff.run(query)


    






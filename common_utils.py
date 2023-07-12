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
from basic_utils import check_content_safety, read_txt
from langchain_utils import create_wiki_tools, create_document_tools, create_google_search_tools, create_custom_llm_agent



chat = ChatOpenAI(temperature=0.0)
embeddings = OpenAIEmbeddings()
delimiter = "####"
delimiter2 = "'''"
delimiter3 = '---'
delimiter4 = '////'

def extract_personal_information(resume):

    # with open(resume_file, 'r', errors='ignore') as f:
    #     resume = f.read()

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

    
    response = chat(messages)
    personal_info_dict = output_parser.parse(response.content)
    print(personal_info_dict.get('email'))
    return personal_info_dict


def fetch_samples(job_title, samples):

    table = find_similar_jobs(job_title)

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


def get_web_resources(query, top=10):

    wiki_tools = create_wiki_tools()
    search_tools = create_google_search_tools(top)
    tools = wiki_tools+search_tools
    # SERPAPI has limited searches, for now, skip
    # tools = create_wiki_tools()
    # agent= initialize_agent(
    #     tools, 
    #     chat, 
    #     # agent=AgentType.REACT_DOCSTORE,
    #     agent="zero-shot-react-description",
    #     handle_parsing_errors=True,
    #     verbose = True,
    #     )
    # agent_executor = AgentExecutor.from_agent_and_tools(
    #     agent=agent.agent,
    #     tools = tools,
    #     verbose = True,
    # )
    agent_executor = create_custom_llm_agent(chat, tools)
    try:
        response = agent_executor.run(query)
        print(f"Success: {response}")
        return response
    except Exception as e:
        response = str(e)
        print(f"Exception: {response}")




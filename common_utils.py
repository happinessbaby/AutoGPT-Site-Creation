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
from langchain.chains import RetrievalQA
from pathlib import Path
from basic_utils import read_txt
from langchain_utils import create_wiki_tools, create_search_tools, create_db_tools, create_QA_chain, create_qa_tools, create_doc_tools, split_doc, retrieve_redis_vectorstore, get_index
from langchain import PromptTemplate
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents.agent_toolkits import create_python_agent
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain.vectorstores import FAISS
import sys
import re



delimiter = "####"
delimiter2 = "'''"
delimiter3 = '---'
delimiter4 = '////'

def extract_personal_information(resume,  llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", cache=False)):

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

def extract_posting_information(posting, llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", cache=False)):
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
    print(posting_info_dict.get('job'))
    return posting_info_dict


def extract_fields(resume, llm=OpenAI(temperature=0, cache=False)):


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
    print(response)
    return response

    
def search_related_samples(job_title, directory):
    # related_jobs = get_completion(f"Generate a list of job titles that are similar to {job_title} or relevant to {job_title}")
    # print(related_jobs)

    system_message = f"""
		You are an assistant that evaluates whether the job position described in the content is related to {job_title} or relevant to {job_title}. 

		Respond with a Y or N character, with no punctuation:
		Y - if the content contains a cover letter
		N - otherwise

		Output a single letter only.
		"""
    related_files = []
    for path in  Path(directory).glob('**/*'):
        # 1. extract job from sample resume/cover letter and do a similarity search with job_title
        # 2. for similar jobs, compare the sample with the user's and determine areas of improvement
        file = str(path)
        content = read_txt(file)
        messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': content}
        ]	
        
        response = get_completion_from_messages(messages, max_tokens=1)
        if (response=="Y"):
            related_files.append(file)
    return related_files




def compare_samples(job_title, query, directory, llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", cache=False)):
    # loader = CSVLoader(file_path="jobs.csv")
    # docs = loader.load()

    # output_parser = CommaSeparatedListOutputParser()
    # qa_stuff = create_QA_chain(llm, embeddings, docs=docs, db_type = "docarray", output_parse = output_parser)

    # # format_instructions = output_parser.get_format_instructions()

    # similarity_query = f"""List all the jobs titles that are similar to {job_title} or relevant to {job_title}.
    
    # Do not make up things not in the given context. """

    # jobs = qa_stuff.run(similarity_query)
    # print(jobs)
    

    # sample_string = ""
    # jobs_list = jobs.split(" ")
    # for job in jobs_list:
    #     job = job[:-1] 
    #     print(job)
    #     if (samples.get(job)!=None):
    #         sample = read_txt(samples.get(job))
    #         sample_string = sample_string + "\n" + f" {delimiter3}\n{sample}\n{delimiter3}" + "\n\nexample:"   
    # print(sample_string)
    # return sample_string
    related_files = search_related_samples(job_title, directory)
    print(related_files)
    if (related_files):
        tools = []
        for file in related_files:
            docs = split_doc(file, "file")
            retriever = FAISS.from_documents(docs, OpenAIEmbeddings()).as_retriever()
            tool = create_db_tools(llm, retriever, Path(file).name )
            tools.extend(tool)
        # Option 1: OpenAI multi functions
        agent = initialize_agent(
            tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
            )
        response = agent.run(query)
        print(response)
        return response
        # Option 2: Plan and Execute
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
        response = agent.run(query)
        print(response)
        return response
    else:
        return ""






def get_web_resources(query, search_tool, top=10,  llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", cache=False)):

    # SERPAPI has limited searches, for now, use plain google
    if (search_tool=="google"):
        tools = create_search_tools("google", top)
    elif (search_tool=="wiki"):
        tools = create_wiki_tools()
    # Option 1: ReACt decent enough, tend to summarize too much 
    agent= initialize_agent(
        tools, 
        llm, 
        # agent=AgentType.REACT_DOCSTORE,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
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
    # Option 2: better at providing details but tend to be very slow and error prone and too many tokens
    planner = load_chat_planner(llm)
    executor = load_agent_executor(llm, tools, verbose=True)
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
    response = agent.run(query)
    print(response)
    return response


    
def get_job_relevancy(doc, query, doctype="file", llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", cache=False)):

    # loader = TextLoader(file_path=resume)
    # docs = loader.load()

    # qa = create_QA_chain(llm, embeddings, docs = docs)

    # tools = create_qa_tools(qa)
    tools = create_doc_tools(doc, doctype)


    agent = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
    )
    response = agent.run(query)
    print(response)
    return response
    

def retrieve_from_db(path, query, llm=OpenAI()):
    index = get_index(path=path, path_type='dir')
    # Create a question-answering chain using the index
    chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=index.vectorstore.as_retriever(), input_key="question")
    response = chain.run(query)
    print(response)
    return response

    
def get_summary(doc_path, llm=OpenAI()):
    docs = split_doc(path=doc_path, path_type="file")
    prompt_template = """Identity the job position, company then provide a summary of the following job posting:

        {text} \n

        Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.

    """
      
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    response = chain.run(docs)
    print(response)
    return response

# def retrieve_from_vectorstore(embeddings, query,  llm=OpenAI(temperature=0, cache=False), db_type = "redis", index_name="redis_resume_advice"):


#     qa_stuff = create_QA_chain(llm, embeddings, db_type=db_type, index_name=index_name)
#     # Option 1: qa tool + react 
#     # Option 1 seems to give better advices 
#     tools = create_qa_tools(qa_stuff)
#     agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
#     try:
#         response = agent.run(query)
#         return response
#     except ValueError as e:
#         response = str(e)
#         if not response.startswith("Could not parse LLM output: `"):
#             print(e)
#             raise e
#         response = response.removeprefix(
#             "Could not parse LLM output: `").removesuffix("`")
#         return response
    # response = agent.run(query)
    # # Option 2
    # # response = qa_stuff.run(query)








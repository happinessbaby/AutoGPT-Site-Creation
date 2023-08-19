import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent, create_json_agent
from basic_utils import read_txt
from common_utils import (get_web_resources, retrieve_from_db, extract_posting_information, get_summary, generate_multifunction_response, create_n_docs_tool,
                           extract_fields, get_field_content, extract_job_title,  search_related_samples, create_sample_tools)
from pathlib import Path
import json
from json import JSONDecodeError
from multiprocessing import Process, Queue, Value
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from typing import Dict, List
import uuid


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
resume_evaluation_path = os.environ["RESUME_EVALUATION_PATH"]
resume_samples_path = os.environ["RESUME_SAMPLES_PATH"]
# TODO: caching and serialization of llm
llm = ChatOpenAI(temperature=0.0, cache=False)
# llm = OpenAI(temperature=0, top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)
embeddings = OpenAIEmbeddings()
# TODO: save these delimiters in json file to be loaded from .env
delimiter = "####"
delimiter1 = "````"
delimiter2 = "////"
delimiter3 = "<<<<"
delimiter4 = "****"



    
def evaluate_resume(my_job_title="", company="", resume_file = "", posting_path="") -> str:
    
    filename = Path(resume_file).stem
    res_path = os.path.join(resume_evaluation_path, filename+".txt")
    generated_responses = {}

    # get resume
    resume = read_txt(resume_file)

    # get resume field names
    resume_fields = extract_fields(resume)
    resume_fields = resume_fields.split(",")
    generated_responses.update({"resume fields": resume_fields})

    job_specification = ""
    # get job specification and company name from job posting link, if provided
    if (Path(posting_path).is_file()):
      job_specification = get_summary(posting_path)
      posting = read_txt(posting_path)
      posting_info_dict=extract_posting_information(posting)
      my_job_title = posting_info_dict["job"]
      company = posting_info_dict["company"]

    # if job title is not provided anywhere else, extract from the resume
    if (my_job_title==""):
      my_job_title = extract_job_title(resume)
    generated_responses.update({"job title": my_job_title})
    generated_responses.update({"company name": company})
    generated_responses.update({"job specification": job_specification})

    # get general job description
    query_job  = f"""Research what a {my_job_title} does, including details of the common skills, responsibilities, education, experience needed for the job."""
    job_description = get_web_resources(query_job)
    generated_responses.update({"job description": job_description})

    # get company description, if provided
    company_description=""
    if (company!=""):
      company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values.                         
                          Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output you don't know"""     
      company_description = get_web_resources(company_query)
    generated_responses.update({"company description": company_description})

    related_samples = search_related_samples(my_job_title, resume_samples_path)
    sample_tools = create_sample_tools(related_samples, "resume")
    relevancy_tools = [create_n_docs_tool]

    # process all fields in parallel
    processes = [Process(target = rewrite_resume_fields, args = (generated_responses, field, resume_file, res_path, relevancy_tools, sample_tools)) for field in resume_fields]

    # start all processes
    for p in processes:
       p.start()

    for p in processes:
       p.join()
   
    # return result to chat agent
    return f"""resume evaluation file": {res_path}"""


def rewrite_resume_fields(generated_response: Dict[str, str], field:str, read_path:str, res_path:str, relevancy_tools:List[Tool], sample_tools:List[Tool]) -> None:
    # field_dict = { }
    print(f"CURRENT FIELD IS: {field}")
    # field_dict[field]= {}
    resume = read_txt(read_path)
    resume_field_content = get_field_content(resume, field)
    my_job_title = generated_response.get("job title", "")
    company_description = generated_response.get("company description", "")
    job_specification = generated_response.get("job specification", "")
    job_description = generated_response.get("job description", "")

    query_missing = f""" 

        Use your tool, compare the {field} in the resume sample documents with the applicant's field content.

        The applicant's field content is delimited with {delimiter} characters.
      
        applicant's field content: {delimiter}{resume_field_content}{delimiter}

        Generalize a list of missing items in the applicant's field content that should be included. 

        If the {field} does not exist in the resume samples, please output -1. 


        """
    missing_items = generate_multifunction_response(query_missing, sample_tools)
            
    query_relevancy = f"""Determine the relevant and irrelevant information contained in the field content. 

      You are provided with job specification for an opening position. 
      
      Use it as a primarily guidelines when generating your answer. 

      You are also provided with a general job decription of the requirement of {my_job_title}. 
      
      Use it as a secondary guideline when forming your answer.

      If job specification is not provided, use general job description as your primarily guideline. 

      field name: {field}

      field content: {resume_field_content}\n

      job specification: {job_specification}\n

      general job description: {job_description} \n

      Generate a list of irrelevant information that should not be included in the field content and a list of relevant information that should be included. 

        """

    relevancy = generate_multifunction_response(query_relevancy, relevancy_tools)

    # Get expert advices 
    advice_query = f"""What are some best practices when writing {field} field for a resume to  make it ATS-friendly?  """
    advices = retrieve_from_db(advice_query)
                                

    template_string = """" Your task is to analyze and help improve the content of resume field {field}. 

    The content of the field is delimiter with {delimiter} characters. Always use this as contenxt and do not make things up. 

    field content: {delimiter}{field_content}{delimiter} \n
    
    Step 1: You're given some expert advices on how to write {field} . Keep these advices in mind for the next steps.

        expert advices: {advices} \n

    Step 2: You're provided with some company informtion, job specification, and job description.

      Try to make resume field cater to the compnay and job position, if possible. 

      company information: {company_description}.  \n

      job specification: {job_specification}.    \n

      job description: {job_description}

    Step 3: You are given two lists of information delimited with {delimiter2} characters. One is content to be included in the {field} and the other is content to be rewritten. 

        information list: {delimiter2}{relevancy}{delimiter2} \n

    Step 4: Rewrite the list of irrelevant content in Step 3 to make them appear more relevant to job position. 

    Step 5: Polish the list of relevant content in Step 3 to make them more ATS-friendly.


    Use the following format:
        Step 1:{delimiter4} <step 1 reasoning>
        Step 2:{delimiter4} <step 2 reasoning>
        Step 3:{delimiter4} <step 3 reasoning>
        Step 4:{delimiter4} <rewrite the irrelevant content>
        Step 5:{delimiter4} <polish the relevant content>


      Make sure to include {delimiter4} to separate every step.
    
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    upgrade_resume_message = prompt_template.format_messages(
        field = field,
        field_content = resume_field_content,
        job = my_job_title,
        advices=advices,
        relevancy = relevancy,
        company_description = company_description,
        job_specification = job_specification, 
        job_description = job_description,
        delimiter = delimiter,
        delimiter2 = delimiter2,
        delimiter4 = delimiter4, 
        
    )

    my_advice = llm(upgrade_resume_message).content
    with open(res_path, 'a') as f:
       f.write(my_advice + "\n" +"[" + missing_items +"]" +"\n")
  




# receptionist
def processing_resume(json_request: str) -> str:
    
    print(json_request)
    try:
      args = json.loads(json_request)
    except JSONDecodeError:
      return "Format in JSON and try again." 
    args = json.loads(json_request)
    # if resume doesn't exist, ask for resume
    if ("resume file" not in args or args["resume file"]=="" or args["resume file"]=="<resume file>"):
      return "Can you provide your resume so I can further assist you? "
    else:
      # may need to clean up the path first
        resume_file = args["resume file"]
    if ("job" not in args or args["job"] == "" or args["job"]=="<job>"):
        job = ""
    else:
       job = args["job"]
    if ("company" not in args or args["company"] == "" or args["company"]=="<company>"):
        company = ""
    else:
        company = args["company"]
    if ("job post link" not in args or args["job post link"]=="" or args["job post link"]=="<job post link>"):
        posting_path = ""
    else:
        posting_path = args["job post link"]

    return evaluate_resume(my_job_title=job, company=company, resume_file=resume_file, posting_path=posting_path)


def create_resume_evaluator_tool() -> List[Tool]:
    
    name = "resume_evaluator"
    parameters = '{{"job":"<job>", "company":"<company>", "resume file":"<resume file>", "job post link": "<job post link>"}}'
    output = '{{"resume evaluation file": "<resume evaluation file>"}}'
    description = f"""Helps to evaluate a resume. Use this tool more than any other tool when user asks to evaluate, review, help with a resume. 
    Do not use this tool if "faiss_resume_advice" tool exists. Use "faiss_resume_advice" instead. 
    Input should be JSON in the following format: {parameters} \n
    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else) 
    Output should be file loaded  from provided format {output}.  \n
    (remember to use file loader tool to load the file to the user)
    """
    tools = [
        Tool(
        name = name,
        func = processing_resume,
        description = description, 
        verbose = False,
        )
    ]
    print("Succesfully created resume evaluator tool.")
    return tools



def test_resume_tool(resume_file="", job="", company="", posting_link="") -> str:
    
    tools = create_resume_evaluator_tool()
    agent= initialize_agent(
        tools, 
        llm=ChatOpenAI(cache=False), 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True,
        )
    response = agent.run(f"""evaluate a resume with following information:
                              job: {job} \n
                              company: {company} \n
                              resume file: {resume_file} \n
                              job post links: {posting_link}\n            
                              """)
    return response
   
   


if __name__ == '__main__':
    my_job_title = 'accountant'
    my_resume_file = 'resume_samples/sample1.txt'
    # evaluate_resume()
    test_resume_tool(resume_file=my_resume_file, job=my_job_title)
 
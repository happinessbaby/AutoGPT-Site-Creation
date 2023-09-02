import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent, create_json_agent
from basic_utils import read_txt
from common_utils import (get_web_resources, retrieve_from_db, extract_posting_information, get_summary, generate_multifunction_response,
                           extract_fields, get_field_content, extract_job_title,  search_related_samples, create_sample_tools, extract_personal_information)
from langchain_utils import create_search_tools
from pathlib import Path
import json
from json import JSONDecodeError
from multiprocessing import Process, Queue, Value
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from typing import Dict, List
from langchain.document_loaders import BSHTMLLoader
import uuid


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
resume_evaluation_path = os.environ["RESUME_EVALUATION_PATH"]
resume_samples_path = os.environ["RESUME_SAMPLES_PATH"]
# TODO: caching and serialization of llm
llm = ChatOpenAI(temperature=0.0)
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
    personal_info_dict = extract_personal_information(resume)
    generated_responses.update(personal_info_dict)

    # get resume field names
    resume_fields = extract_fields(resume)
    resume_fields = resume_fields.split(",")
    # generated_responses.update( {"resume_fields)

    job_specification = ""
    # get job specification and company name from job posting link, if provided
    if (Path(posting_path).is_file()):
      prompt_template = """Identity the job position, company then provide a summary of the following job posting:
        {text} \n
        Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
      """
      job_specification = get_summary(posting_path, prompt_template)
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
    #TODO: this can be a general putpose tool with all web data  
    general_tools = create_search_tools("google", 3)
    # relevancy_tools = [search_relevancy_advice]

    for field in resume_fields:
      improve_resume_fields(generated_responses, field, resume, res_path, general_tools, sample_tools)

    # process all fields in parallel
    # processes = [Process(target = improve_resume_fields, args = (generated_responses, field, resume, res_path, relevancy_tools, sample_tools)) for field in resume_fields]

    # start all processes
    # for p in processes:
    #    p.start()

    # for p in processes:
    #    p.join()
   
    # return result to chat agent
    return f"""resume evaluation file": {res_path}"""


def improve_resume_fields(generated_response: Dict[str, str], field:str, resume:str, res_path:str, general_tools:List[Tool], sample_tools:List[Tool]) -> None:
    
    print(f"CURRENT FIELD IS: {field}")
    resume_field_content = get_field_content(resume, field)
    my_job_title = generated_response.get("job title", "")
    company_description = generated_response.get("company description", "")
    job_specification = generated_response.get("job specification", "")
    job_description = generated_response.get("job description", "")
    education_level = generated_response.get("education", "")

    # The purpose of getting expert advices is to evaluate weaknesses in the current resume field content
    advice_query = f"""ATS-friendly way of writing {field}"""
    advices = retrieve_from_db(advice_query)
    query_evaluation = f"""You are given some expert advices on writing {field} section of the resume.

    Use there advices to identity the strenghts and weaknesses of the resume content delimited with {delimiter} characters.
    
    field content: {delimiter}{resume_field_content}{delimiter}
    """
    strengh_weakness = generate_multifunction_response(query_evaluation, general_tools)

    # The purpose of this missing query is to identity any missing details that otherwise should be included in the resume field
    query_missing = f""" 

        Use your tool, compare the {field} in the resume sample documents with the applicant's field content.

        The applicant's field content is delimited with {delimiter} characters.
      
        applicant's field content: {delimiter}{resume_field_content}{delimiter}

        Generalize a list of missing items in the applicant's field content that should be included. 

        If the {field} does not exist in any of the resume samples, please output -1. 

        """
    missing_items = generate_multifunction_response(query_missing, sample_tools)

    # Get resume's relevant and irrelevant info for resume field: few-shot learning works great here
    # The purpose of identitying irrelevant and relevant information is so that irrelevant information can be deleted or reworded   
    query_relevancy = f"""Determine the relevant and irrelevant information contained in the field content. 

     Generate a list of irrelevant information that should not be included in the cover letter and a list of relevant information that should be included in the resume field.
       
      Remember to use either job specification or general job description as your guideline. Don't forget to use your tool "search_relevancy_advice".

      field name: {field}

      field content: {resume_field_content}\n

      job specification: {job_specification}\n

      general job description: {job_description} \n

      Your answer should be detailed and only from the resume. Please also provide your reasoning too. 
        
      For example, your answer may look like this:

      Relevant information in the Experience field:

      1. Experience as a Big Data Engineer: skills and responsibilities of a Big Data Engineer are transferable to those of a data analyst

      Irrelevant information  in the Experience Field:

      1. Experience as a front desk receptionist is not directly related to the role of a data analyst

      Please do not do anything and output -1 if field name is personal informaion or contains personal information such as name and links. These do not need any evaluation on relevancy

      """

    relevancy = generate_multifunction_response(query_relevancy, general_tools)


    template_string = """ You are a professional resume writer. Your task is to polish some poorly written resume fields according to another professional's report on what should be improved. 

    The content of the field is delimited with {delimiter} characters.

    field content: {delimiter}{field_content}{delimiter}
    
    Here is the report on what should be improved: {missing_items} + {relevancy} \n

    Here are some advices on how to improve them: {advices} \n
     
    Please rewrite the resume field. Output only the rewritten content.  """

    template_string = """ Please report the following information to the user in a detailed and well-formatted manner. 

    {missing_items} + {relevancy} 
    """


    prompt_template = ChatPromptTemplate.from_template(template_string)
    upgrade_resume_message = prompt_template.format_messages(
        field_content = resume_field_content,
        advices=advices,
        relevancy = relevancy,
        missing_items = missing_items,
        delimiter = delimiter,     
    )

    my_advice = llm(upgrade_resume_message).content

    print(my_advice)
    with open(res_path, 'a') as f:
       f.write(my_advice)
  



# receptionist
def processing_resume(json_request: str) -> str:   
       
    """ Input parser: input is LLM's action_input in JSON format. This function then processes the JSON data and feeds them to the resume evaluator. """

    try:
      json_request = json_request.strip("'<>() ").replace('\'', '\"')
      args = json.loads(json_request)
    except JSONDecodeError:
      return "Format in JSON and try again." 

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
    
    """ Input parser: input is user's input as a string of text. This function takes in text and parses it into JSON format. 
    
    Then it calls the processing_resume function to process the JSON data. """
    
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


   


if __name__ == '__main__':
    my_job_title = 'accountant'
    my_resume_file = 'resume_samples/sample1.txt'
    # evaluate_resume()

 
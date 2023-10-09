from basic_utils import read_txt
from openai_api import get_completion
from openai_api import get_completion, get_completion_from_messages
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate,  StringPromptTemplate
from langchain.agents import load_tools, initialize_agent, Tool, AgentExecutor
from langchain.document_loaders import CSVLoader, TextLoader
from pathlib import Path
from basic_utils import read_txt, convert_to_txt
from langchain_utils import create_search_tools, generate_multifunction_response
from langchain import PromptTemplate
from common_utils import get_generated_responses, get_web_resources
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from typing import Any, List, Union, Dict
from langchain.docstore.document import Document
from langchain.tools import tool
import json
from json import JSONDecodeError
import faiss
import asyncio
import random
import base64
from datetime import date
from feast import FeatureStore
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


def customize_personal_statement(personal_statement="", resume="", about_me="", posting_path=""):

    personal_statement = read_txt(personal_statement)
    try: 
        resume_content = read_txt(resume)
    except Exception:
        resume_content = ""
    generated_info_dict=get_generated_responses(resume_content = resume_content, about_me=about_me, posting_path=posting_path)

    institution_description=generated_info_dict.get("institution description", "")
    program_description = generated_info_dict.get("program description", "")

    query = f""" You are to help Human polish a personal statement. 
    Prospective employers and universities may ask for a personal statement that details qualifications for a position or degree program.
    You are provided with a personal statement and various pieces of information, if available, that you may use.
    personal statement: {personal_statement}
    institution description: {institution_description}
    program description: {program_description}

    please consider blending these information into the existing personal statement. 
    Use the same tone and style when inserting information.
    Correct grammar and spelling mistakes. 

    """
    tools = create_search_tools("google", 3)
    response = generate_multifunction_response(query, tools)
    return response

def customize_cover_letter(cover_letter="", resume="", about_me="", posting_path="", llm = ChatOpenAI(temperature=0.5, cache=False)):

    cover_letter = read_txt(cover_letter)
    try:
        resume_content = read_txt(resume)
    except Exception:
        resume_content = ""
    generated_info_dict=get_generated_responses(resume_content = resume_content, about_me=about_me, posting_path=posting_path)
    
    job_description = generated_info_dict.get("job description", "")
    job_specification = generated_info_dict.get("job specification", "")
    about_me = generated_info_dict.get("about me", "")
    company_description = generated_info_dict.get("company description", "")
    company = generated_info_dict.get("company", "")
    job = generated_info_dict.get("job", "")
    field_names = generated_info_dict.get("field names")
    
    query = f""" Your task is to help Human revise and polish a cover letter. You are provided with a cover letter and various pieces of information, if available, that you may use.
    Foremost important is the applicant's about me that you should always keep in mind.
    The applicant is applying for job {job} at company {company}. Please change the cover letter's existing information to suit the purpose of applying to this job and this company, if available.
    Please consider blending the below information into the existing cover letter to make it more descriptive, personalized, and appropriate for the occasion.

    about me: {about_me}
    cover letter: {cover_letter}
    job desciption: {job_description}
    job specification: {job_specification}
    company description: {company_description}

    Use the same tone and style when inserting information.
    Correct grammar and spelling mistakes. 

    """
    # template_string = """ Your task is to help Human revise and polish a cover letter. You are provided with a cover letter and various pieces of information, if available, that you may use.
    # Foremost important is Human's about me that you should always keep in mind.
    # about me: {about_me}
    # cover letter: {cover_letter}
    # job desciption: {job_description}
    # job specification: {job_specification}
    # company description: {company_description}
    # today'date: {date}

    # please consider blending these information into the existing cover letter. 
    # Use the same tone and style when inserting information.
    # Correct grammar and spelling mistakes. 

    # """
    tools = create_search_tools("google", 3)
    response = generate_multifunction_response(query, tools, early_stopping=True)
    # prompt_template = ChatPromptTemplate.from_template(template_string)
    # cover_letter_message = prompt_template.format_messages(
    #     cover_letter = cover_letter,
    #     about_me = about_me,
    #     date = date.today(),
    #     company_description = company_description, 
    #     job_description = job_description,
    #     job_specification = job_specification,

    # )

    # response = llm(cover_letter_message).content
    print(response)
    return response

def customize_resume(resume="", about_me="", posting_path=""):

    resume_content = read_txt(resume)
    generated_info_dict=get_generated_responses(resume_content = resume_content, about_me=about_me, posting_path=posting_path)
    job_description = generated_info_dict.get("job description", "")
    job_specification = generated_info_dict.get("job specification", "")
    about_me = generated_info_dict.get("about me", "")
    company_description = generated_info_dict.get("company description", "")
    field_names = generated_info_dict.get("field names", "")

    query = f""" Your task is to help Human revise and polish a resume. You are provided with a resume and various pieces of information, if available, that you may use.
    Foremost important is Human's about me that you should always keep in mind.
    about me: {about_me}
    resume: {resume_content}
    job desciption: {job_description}
    job specification: {job_specification}
    company description: {company_description}

    please consider blending these information into the existing resume. 
    Use the same tone and style when inserting information.
    Correct grammar and spelling mistakes. 

    """
    tools = create_search_tools("google", 3)
    response = generate_multifunction_response(query, tools)
    return response




def create_resume_customize_writer_tool() -> List[Tool]:

    """ Agent tool that calls the function that customizes resume. """

    name = "resume_customize_writer"
    parameters = '{{ "job post link":"<job post link>", "about me":"<about me>", "resume file":"<resume file>"}}'
    description = f""" Helps customize and personalize resume. 
    Input should be a single string strictly in the following JSON format: {parameters}
    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else)"""
    tools = [
        Tool(
        name = name,
        func = process_resume,
        description = description, 
        verbose = False,
        handle_tool_error=True,
        )
    ]
    print("Succesfully created resume customize wrtier tool.")
    return tools

def create_cover_letter_customize_writer_tool() -> List[Tool]:

    """ Agent tool that calls the function that customizes cover letter. """

    name = "cover_letter_customize_writer"
    parameters = '{{"cover letter file":"<cover letter file>", "about me":"<about me>", "job post link:"<job post link>",  "resume file":"<resume file>"}}'
    description = f""" Helps customize, personalize, and improve cover letter.
    Input should be a single string strictly in the following JSON format: {parameters}
    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else)"""
    tools = [
        Tool(
        name = name,
        func = process_cover_letter,
        description = description, 
        verbose = False,
        handle_tool_error=True,
        )
    ]
    print("Succesfully created cover letter customize writer tool.")
    return tools

def create_personal_statement_customize_writer_tool() -> List[Tool]:

    """ Agent tool that calls the function that customizes personal statement """

    name = "personal_statement_customize_writer"
    parameters = '{{"personal statement file":"<personal statement file>", "about me":"<about me>", "job post link:"<job post link>",  "resume file":"<resume file>"}}'
    description = f""" Helps customize, personalize, and improve personal statement.
    Input should be a single string strictly in the following JSON format: {parameters}
    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else)"""
    tools = [
        Tool(
        name = name,
        func = process_personal_statement,
        description = description, 
        verbose = False,
        handle_tool_error=True,
        )
    ]
    print("Succesfully created personal statement customize writer tool.")
    return tools

def process_cover_letter(json_request:str) -> str:

    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    
    if ("cover letter file" not in args or args["cover letter file"]=="" or args["cover letter file"]=="<cover letter file>"):
        return """stop using or calling the cover_letter_customize_writer tool. Ask user to upload their cover letter instead. """
    else:
        cover_letter = args["cover letter file"]
    if ("about me" not in args or args["about me"] == "" or args["about me"]=="<about me>"):
        about_me = ""
    else:
        about_me = args["about me"]
    if ("job post link" not in args or args["job post link"]=="" or args["job post link"]=="<job post link>"):
        posting_path = ""
    else:
        posting_path = args["job post link"]
    if ("resume file" not in args or args["resume file"]=="" or args["resume file"]=="<resume file>"):
        resume = ""
    else:
        resume = args["resume file"]

    return customize_cover_letter(cover_letter=cover_letter, resume=resume, about_me=about_me, posting_path=posting_path)

def process_personal_statement(json_request:str) -> str:

    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    
    if ("personal statement file" not in args or args["personal statement file"]=="" or args["personal statement file"]=="<personal statement file>"):
        return """stop using or calling the personal_statement_customize_writer tool. Ask user to upload their personal statement instead. """
    else:
        personal_statement = args["personal statement file"]
    if ("about me" not in args or args["about me"] == "" or args["about me"]=="<about me>"):
        about_me = ""
    else:
        about_me = args["about me"]
    if ("job post link" not in args or args["job post link"]=="" or args["job post link"]=="<job post link>"):
        posting_path = ""
    else:
        posting_path = args["job post link"]
    if ("resume file" not in args or args["resume file"]=="" or args["resume file"]=="<resume file>"):
        resume = ""
    else:
        resume = args["resume file"]

    return customize_personal_statement(personal_statement=personal_statement, resume=resume, about_me=about_me, posting_path=posting_path)




def process_resume(json_request: str) -> str:

    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    
    
    if ("resume file" not in args or args["resume file"]=="" or args["resume file"]=="<resume file>"):
        return """stop using or calling the resume_customize_writer tool. Ask user to upload their resume instead. """
    else:
        resume = args["resume file"]
    if ("about me" not in args or args["about me"] == "" or args["about me"]=="<about me>"):
        about_me = ""
    else:
        about_me = args["about me"]
    if ("job post link" not in args or args["job post link"]=="" or args["job post link"]=="<job post link>"):
        posting_path = ""
    else:
        posting_path = args["job post link"]

    return customize_resume(resume=resume, about_me=about_me, posting_path=posting_path)


if __name__=="__main__":
    personal_statement = "./uploads/file/personal_statement.txt"
    resume = "./resume_samples/resume2023v3.txt"
    posting_path = "./uploads/link/software05.txt"
    # customize_personal_statement(about_me="i want to apply for a MSBA program at university of louisville", personal_statement = personal_statement)
    cover_letter = "cv01.txt"
    customize_cover_letter(about_me = "", cover_letter=cover_letter, resume=resume, posting_path=posting_path)
    # customize_resume (about_me="", resume=resume, posting_path=posting_path)
        
import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent, create_json_agent
from basic_utils import read_txt
from common_utils import (get_web_resources, retrieve_from_db, extract_posting_information, extract_education_level, extract_work_experience_level, get_generated_responses,
                           extract_resume_fields, extract_job_title,  search_related_samples, create_sample_tools, extract_personal_information)
from langchain_utils import create_search_tools, create_mapreduce_chain, create_summary_chain, generate_multifunction_response, create_refine_chain, handle_tool_error
from pathlib import Path
import json
from json import JSONDecodeError
from multiprocessing import Process, Queue, Value
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from typing import Dict, List, Optional
from langchain.document_loaders import BSHTMLLoader
from langchain.tools import tool
from langchain.agents.agent_toolkits import FileManagementToolkit
import docx
import uuid


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
# resume_evaluation_path = os.environ["RESUME_EVALUATION_PATH"]
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


    dirname, fname = os.path.split(resume_file)
    filename = Path(fname).stem +"_re"+".txt"

    # get resume info
    resume_content = read_txt(resume_file)
    info_dict=get_generated_responses(resume_content, my_job_title, company, posting_path)
    highest_education_level = info_dict["highest education level"]
    work_experience_level = info_dict["work experience level"]
    field_names = info_dict["field names"]


    # # Note: retrieval query works best if clear, concise, and detail-dense
    advice_query = f"""what to include for resume with someone with {highest_education_level} and {work_experience_level} for {my_job_title}"""
    advices = retrieve_from_db(advice_query)
    # samples query to find missing content from resume using document comparison
    # Note: document comparison benefits from a clear and simple prompt
    related_samples = search_related_samples(my_job_title, resume_samples_path)
    sample_tools, tool_names = create_sample_tools(related_samples, "resume")
    query_missing = f"""  You are an expert resume field advisor that specializes in improvement content of resume delimited with {delimiter} characters. 

        resume: {delimiter}{resume_content}{delimiter}. 

    Step 1. Sample resume are provided in your tools. Research {str(tool_names)} and answer the following question:

       List common things these resume have in common. 

       Please be general with your answer and ignore any personal details, locations, names, dates, etc. 

    Common Content: 

    Step 2: Your job is to make a list of missing content in the resume delimited with {delimiter} characters using Common Content from Step 1 and  Expert Advice provided below. 

        Expert Advice: {advices}
     
        Ignore all formatting advice and ignore any personal details, dates, school and company names, etc in Common Content. 

        Your answer should be general enough to allow applicant to fill out the missing content with their own information. It is okay if there is nothing missing.

        Missing Content:

        Remember to reference Common Content and Expert Advice when generating Missing Content. Be general enough for applicant to provide their own missing information. 
     """

    missing_items = generate_multifunction_response(query_missing, sample_tools)
    # file_path = file_path = os.path.join(dirname, "missing.txt")
    # with open(res_path, "w") as f:
    #   f.write(missing_items)
    
    # files = []
    for field in field_names:
      # file_path = os.path.join(dirname, f"{field}.txt")
      try: 
        field_content = info_dict[field]
        improve_resume_fields(info_dict, field, field_content,  sample_tools, filename, dirname)
      except Exception as e:
         raise e

      # files.append(file_path)

    # main_template = """Write a detailed summary of the following: {text}
    #     DETAILED SUMMARY: """
    # refine_template = (
    #     "Your job is to produce a final summary\n"
    #     "We have provided an existing summary up to a certain point: {existing_answer}\n"
    #     "We have the opportunity to refine the existing summary"
    #     "(only if needed) with some more context below.\n"
    #     "------------\n"
    #     "{text}\n"
    #     "------------\n"
    #     "Given the new context, refine the original summary."
    #     "If the context isn't useful, return the original summary."
    # )
    # response = create_refine_chain(files, main_template, refine_template)
    # return response
    # return create_mapreduce_chain(files)






    # process all fields in parallel
    # processes = [Process(target = improve_resume_fields, args = (generated_responses, field, resume, res_path, relevancy_tools, sample_tools)) for field in resume_fields]

    # start all processes
    # for p in processes:
    #    p.start()

    # for p in processes:
    #    p.join()

    # return result to chat agent
    return f""" file_path: "missing.txt" """


def improve_resume_fields(generated_response: Dict[str, str], field: str, field_content: str, tools: List[Tool], filename: str, dirname: str) -> None:

    print(f"CURRENT FIELD IS: {field}")
    my_job_title = generated_response.get("job title", "")
    company_description = generated_response.get("company description", "")
    job_specification = generated_response.get("job specification", "")
    job_description = generated_response.get("job description", "")
    # education_level = generated_response.get("education", "")

    # The purpose of getting expert advices is to evaluate weaknesses in the current resume field content
    advice_query =  f"""What are some dos and don'ts when writing resume field {field} to make it ATS-friendly?"""
    advices = retrieve_from_db(advice_query)
    query_evaluation = f""" You are an expert resume advisor. You are given some expert advices on writing {field} section of the resume.
    
    advices: {advices}

    Use these advices to identity the strenghts and weaknesses of the resume content delimited with {delimiter} characters. 
    
    field content: {delimiter}{field_content}{delimiter}. \n
    
    Please provide proof of your reasoning. For example, if there is gap in work history, mark it as a weakness unless there is evidence it should not be considered so.
    
    Ignore all formatting. They should not be considered as weaknesses or strengths. 

    """
    strength_weakness = generate_multifunction_response(query_evaluation, tools)

    # Get resume's relevant and irrelevant info for resume field: few-shot learning works great here
    # The purpose of identitying irrelevant and relevant information is so that irrelevant information can be deleted or reworded
    query_relevancy = f"""You are an expert resume advisor. Determine the relevant and irrelevant information contained in the field content. 

     Generate a list of irrelevant information that should not be included in the resume field and a list of relevant information that should be included in the resume field.
       
      Remember to use either job specification or general job description as your guideline. Don't forget to use your tool "search_relevancy_advice".

      field name: {field}

      field content: {field_content}\n

      job specification: {job_specification}\n

      general job description: {job_description} \n

      Your answer should be detailed and only from the resume. Please also provide your reasoning too. 
        
      For example, your answer may look like this:

      Relevant information in the Experience field:

      1. Experience as a Big Data Engineer: skills and responsibilities of a Big Data Engineer are transferable to those of a data analyst

      Irrelevant information  in the Experience Field:

      1. Experience as a front desk receptionist is not directly related to the role of a data analyst

      You do not need to evaluate field that contains personal information. 

      """

    relevancy = generate_multifunction_response(query_relevancy, tools)


    query = f""" You are a professional resume proof reader. 
    
    Your task is to rewrite the resume field content based on an evaluation on its strengths, weaknesses, and what should be and should not be included, which has been provided for you.

    The original field is: {field_content}

    Its strengths and weakness: {strength_weakness}

    What to include and what not to include: {relevancy}

    Rememeber, your task is to proof-read and rewrite, not reporting information to the user.

    Revised version:  

    Please use your tool "write_file" to write the final revised version to path: {filename}
    
    """

    working_directory=dirname
    write_tool = FileManagementToolkit(
          root_dir=working_directory, # ensures only the working directory is accessible 
          selected_tools=["write_file"],
      ).get_tools()
  
    response = generate_multifunction_response(query, write_tool)

    # with open(file_path, "a") as f:
    #    f.write(strength_weakness + '\n' + relevancy + '\n')



# @tool("resume evaluator")
# def resume_evaluator_tool(resume_file: str, job: Optional[str]="", company: Optional[str]="", job_post_link: Optional[str]="") -> str:

#    """Evaluate a resume when provided with a resume file, job, company, and/or job post link.
#         Note only the resume file is necessary. The rest are optional.
#         Use this tool more than any other tool when user asks to evaluate, review, help with a resume. """

#    return evaluate_resume(my_job_title=job, company=company, resume_file=resume_file, posting_path=job_post_link)
      


@tool(return_direct=True)
def resume_evaluator(json_request: str)-> str:

    """Helps to evaluate a resume. Use this tool more than any other tool when user asks to evaluate, review, help with a resume. 

    Input should be  a single string strictly in the following JSON format:  '{{"job":"<job>", "company":"<company>", "resume file":"<resume file>", "job post link":"<job post link>"}}' \n

    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else) \n

     
    """

    try:
      json_request = json_request.strip("'<>() ").replace('\'', '\"')
      args = json.loads(json_request)
    except JSONDecodeError as e:
      print(f"JSON DECODE ERROR: {e}")
      return "Format in a single string JSON and try again."


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







def processing_resume(json_request: str) -> str:

    """ Input parser: input is LLM's action_input in JSON format. This function then processes the JSON data and feeds them to the resume evaluator. """

    try:
      json_request = json_request.strip("'<>() ").replace('\'', '\"')
      args = json.loads(json_request)
    except JSONDecodeError as e:
      print(f"JSON DECODER ERROR: {e}")
      return "Format in JSON and try again."

    # if resume doesn't exist, ask for resume
    if ("resume file" not in args or args["resume file"]=="" or args["resume file"]=="<resume file>"):
      return "Stop using the resume evaluator tool. Ask user for their resume."
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
    output = '{{"file_path": "<file_path>"}}'
    description = f"""Helps to evaluate a resume. Use this tool more than any other tool when user asks to evaluate, review, help with a resume. 
    Input should be a single string strictly in the following JSON format: {parameters} \n
     Leave value blank if there's no information provided. DO NOT MAKE STUFF UP. 
     (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else) \n
    Output should be calling the 'read_file' tool in the followng JSON format: {output} \n
    The file_path is in the working directory. Output the content directly to user please.' \n 
    """
    tools = [
        Tool(
        name = name,
        func = processing_resume,
        description = description,
        verbose = False,
        handle_tool_error=True,

        )
    ]
    print("Succesfully created resume evaluator tool.")
    return tools





if __name__ == '__main__':
    my_job_title = 'accountant'
    my_resume_file = 'resume_samples/sample1.txt'
    # evaluate_resume()



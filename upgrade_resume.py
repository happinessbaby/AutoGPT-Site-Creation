import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent, create_json_agent
from basic_utils import read_txt, write_to_docx_template
from common_utils import (get_web_resources, retrieve_from_db,  extract_education_level, extract_work_experience_level, get_generated_responses,
                           extract_resume_fields, search_related_samples, create_sample_tools, extract_personal_information)
from langchain_utils import create_search_tools, create_mapreduce_chain, create_summary_chain, generate_multifunction_response, create_refine_chain, handle_tool_error
from pathlib import Path
import json
from json import JSONDecodeError
from multiprocessing import Process, Queue, Value
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits import JsonToolkit
from typing import Dict, List, Optional
from langchain.document_loaders import BSHTMLLoader
from langchain.tools import tool
from langchain.agents.agent_toolkits import FileManagementToolkit
import docx
import uuid
from docxtpl import DocxTemplate	
from docx import Document
from docx.shared import Inches


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

# these are resume template file and template field names
# doc = DocxTemplate("./resume_templates/template3.docx")
# personal_info = ["Name", "Phone", "Email", "LinkedIn", "Website", "JobTitle"]
document = Document()
document.add_heading('Resume Evaluation', 0)
document_dict = {}


def evaluate_resume(about_me="", resume_file = "", posting_path="") -> str:


    dirname, fname = os.path.split(resume_file)
    filename = Path(fname).stem 
    docx_filename = filename + "_evaluation"+".docx"
    # get resume info
    resume_content = read_txt(resume_file)
    info_dict=get_generated_responses(resume_content=resume_content, posting_path=posting_path, about_me=about_me)
    highest_education_level = info_dict.get("highest education level", "")
    work_experience_level = info_dict.get("work experience level", "")
    name = info_dict.get("name", "")
    phone = info_dict.get("phone", "")
    email = info_dict.get("email", "")
    linkedin = info_dict.get("linkedin", "")
    website = info_dict.get("website", "")
    job = info_dict.get("job", "")
    company = info_dict.get("company", "")
    field_names = info_dict.get("field names", "")
    personal_info_dict = {"Name":name, "Phone":phone, "Email":email, "LinkedIn": linkedin, "Website": website, "JobTitle":job}
    print(personal_info_dict) 
    # write_to_docx_template(doc, personal_info, personal_info_dict, docx_filename)

    # Note: document comparison benefits from a clear and simple prompt
    related_samples = search_related_samples(job, resume_samples_path)
    sample_tools, tool_names = create_sample_tools(related_samples, "resume")

    impression_query = f""" You are an expert resume critic who assesses the quality of a resume. 

    Answer the following question: 

        what is your overall impression of the applicant's resume delimiter with {delimiter} characters below? 
                              
    applicant's resume: {delimiter}{resume_content}{delimiter} \n

   Please generate your impression for the applicant's resume only and write it in one paragraph. 
   
   Be critical of the length, the quality of writing, and the extensiveness of the fields.

   If you believe there to be any missing fields, please inform Human. 
   
   Use your tools if you need to reference other resume.

   """
    # # Note: retrieval query works best if clear, concise, and detail-dense
    # samples query to find missing content from resume using document comparison
    impression = generate_multifunction_response(impression_query, sample_tools)
    document_dict.update({"Overall Impression":impression})
    # document.add_heading(f"Overall Impression", level=2)
    # document.add_paragraph(impression)
    # document.add_page_break()

    advice_query = f"""what fields to include for resume with someone with {highest_education_level} and {work_experience_level} for {job}"""
    advice = retrieve_from_db(advice_query)

    query_missing = f"""  
  
    Generate a list of missing resume fields suggestions in the job applicant's resume fields given the expert advice. 

    expert advice: {advice}

    job applicant's resume fields: {str(field_names)}

    If you believe the applicant's resume fields are enough, output -1. 

    Use your tools if you need to reference other resume.

    """
    missing_fields = generate_multifunction_response(query_missing, sample_tools)
    document_dict.update({"Possible Missing Fields": missing_fields})
    # document.add_heading(f"Possible Missing Fields", level=2)
    # document.add_paragraph(missing_fields)
    # document.add_page_break()
    # document.save(docx_filename)

    # for field in field_names:
    #   if field != "Personal Information":
    #     try: 
    #       field_content = info_dict[field]
    #       improve_resume_fields(info_dict, field, field_content,  sample_tools, docx_filename)
    #     except Exception as e:
    #       raise e
      
    # document.save(docx_filename)
    # print("Successfully saved resume evaluation.")

    # process all fields in parallel
    processes = [Process(target = improve_resume_fields, args = (info_dict, field, info_dict.get(field, ""),  sample_tools, document_dict)) for field in field_names]
    for p in processes:
       p.start()
    for p in processes:
       p.join()

    return document_dict

    # return result to chat agent
    # return f""" file_path: {docx_filename} """


def improve_resume_fields(generated_response: Dict[str, str], field: str, field_content: str, tools: List[Tool], document_dict: Dict[str, str]) -> None:

    print(f"CURRENT FIELD IS: {field}")
    job = generated_response.get("job", "")
    company_description = generated_response.get("company description", "")
    job_specification = generated_response.get("job specification", "")
    job_description = generated_response.get("job description", "")
    highest_education_level = generated_response.get("highest education level", "")
    work_experience_level = generated_response.get("work experience level", "")
    # education_level = generated_response.get("education", "")

    advice_query =  f"""how to make resume field {field} ATS-friendly? No formatting advices."""
    advice1 = retrieve_from_db(advice_query)
    advice_query = f"""what to include in {field} of resume for {highest_education_level} and {work_experience_level} as a {job}"""
    advice2 = retrieve_from_db(advice_query)

    query_evaluation = f"""  You are an expert resume field advisor. 

     Generate a list of missing, irrelevant, and not ATS-friendly information in the resume field content. 
       
     Remember to use either job specification or general job description as your guideline along with the expert advice.

      field name: {field}

      field content: {field_content}\n

      job specification: {job_specification}\n

      general job description: {job_description} \n

      expert advice: {advice2} + "\n" + {advice1}

      Your answer should be detailed and only from the field content. Please also provide your reasoning too as in the following examples:

            Missing or Irrelevant Field Content for Work Experience:

            1. Quantative achievement is missing: no measurable metrics or KPIs to highlight any past achievements. 

            2. Front desk receptionist is irrelevant: Experience as a front desk receptionist is not directly related to the role of a data analyst

            3. Date formatting is not ATS-friendly: an ATS-friendly way to write dates is for example, 01/2001 or January 2001
    
      Please ignore all formatting advices as formatting should not be part of the assessment.

      Use your tools if you need to reference other resume.

     """
    evaluation = generate_multifunction_response(query_evaluation, tools)
    document_dict.update({f"{field} evaluation":evaluation})
    

    # document.add_heading(f"{field}", level=1)
    # document.add_paragraph(evaluation)
    # document.add_page_break()
    # document.save(docx_filename)





    # The purpose of getting expert advices is to evaluate weaknesses in the current resume field content
    # advice_query =  f"""What are some dos and don'ts when writing resume field {field} to make it ATS-friendly?"""
    # advices = retrieve_from_db(advice_query)
    # query_evaluation = f"""You are given some expert advices on writing {field} section of the resume most ATS-friendly.
    
    # expert advices: {advices}

    # Use these advices to generate a list of weaknesses of the field content and a list of the strengths.
    
    # field content: {field_content}. \n

    # Your answer should be detailed and only from the field content. Please also provide your reasoning too as in the following examples:
    
    #     Weaknesses of Objective:

    #     1. Weak nouns: you wrote  “LSW” but if the ATS might be checking for “Licensed Social Worker,”

    #     Strengths of Objective:

    #     1. Strong adjectives: "Dedicated" and "empathetic" qualities make a good social worker

    # Please ignore all formatting advices as formatting should not be part of the assessment.

    # Use your tools if you need to reference other resume.

    # """
    # vulnerability = generate_multifunction_response(query_evaluation, tools)

    # # Get resume's relevant and irrelevant info for resume field: few-shot learning works great here
    # # The purpose of identitying irrelevant and relevant information is so that irrelevant information can be deleted or reworded
    # query_relevancy = f"""You are an expert resume advisor. Determine the relevant and irrelevant information contained in the field content. 

    #  Generate a list of irrelevant information that should not be included in the resume field and a list of relevant information that should be included in the resume field.
       
    #   Remember to use either job specification or general job description as your guideline. Don't forget to use your tool "search_relevancy_advice".

    #   field name: {field}

    #   field content: {field_content}\n

    #   job specification: {job_specification}\n

    #   general job description: {job_description} \n

    #   Your answer should be detailed and only from the resume. Please also provide your reasoning too as in the following example:

    #       Relevant information in the Experience field:

    #       1. Experience as a Big Data Engineer: skills and responsibilities of a Big Data Engineer are transferable to those of a data analyst

    #       Irrelevant information  in the Experience Field:

    #       1. Experience as a front desk receptionist is not directly related to the role of a data analyst
    
    #   Please ignore all formatting advices as formatting should not be part of the assessment.

    #   Use your tools if you need to reference other resume.

    #   """
    
    # relevancy = generate_multifunction_response(query_relevancy, tools)
        
    # query_missing_field = f"""  You are an expert resume field advisor. 

    #  Generate a list of missing information that should be included in the resume field content. 
       
    #  Remember to use either job specification or general job description and comany description as your guideline. 

    #   field name: {field}

    #   field content: {field_content}\n

    #   job specification: {job_specification}\n

    #   general job description: {job_description} \n

    #   company description: {company_description}

    #   Your answer should be detailed and only from the field content. Please also provide your reasoning too as in the following examples:

    #         Missing Field Content for Work Experience:

    #         1. Quantative achievement in project management: no measurable metrics or KPIs to highlight any past achievements. 

    #   Please ignore all formatting advices as formatting should not be part of the assessment.

    #   Use your tools if you need to reference other resume.

    #  """
    # missing = generate_multifunction_response(query_missing_field, tools)

    # evaluation = vulnerability + '\n' + relevancy + '\n' + missing
    # document.add_heading(f"{field}", level=1)
    # document.add_paragraph(evaluation)
    # document.add_page_break()
    # document.save(docx_filename)








# @tool("resume evaluator")
# def resume_evaluator_tool(resume_file: str, job: Optional[str]="", company: Optional[str]="", job_post_link: Optional[str]="") -> str:

#    """Evaluate a resume when provided with a resume file, job, company, and/or job post link.
#         Note only the resume file is necessary. The rest are optional.
#         Use this tool more than any other tool when user asks to evaluate, review, help with a resume. """

#    return evaluate_resume(my_job_title=job, company=company, resume_file=resume_file, posting_path=job_post_link)
      


@tool(return_direct=True)
def resume_evaluator(json_request: str)-> str:

    """Helps to evaluate a resume. Use this tool more than any other tool when user asks to evaluate or review a resume. 

    Input should be  a single string strictly in the following JSON format:  '{{"about me":"<about me>", "resume file":"<resume file>", "job post link":"<job post link>"}}' \n

    Leave value blank if there's no information provided. DO NOT MAKE STUFF UP. 

     (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else) \n

     Output should be using the "get download link" tool in the following single string JSON format: '{{"file_path":"<file_path>"}}'
   
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
    if ("about me" not in args or args["about me"] == "" or args["about me"]=="<about me>"):
        about_me = ""
    else:
        about_me = args["about me"]
    # if ("job" not in args or args["job"] == "" or args["job"]=="<job>"):
    #     job = ""
    # else:
    #    job = args["job"]
    # if ("company" not in args or args["company"] == "" or args["company"]=="<company>"):
    #     company = ""
    # else:
    #     company = args["company"]
    if ("job post link" not in args or args["job post link"]=="" or args["job post link"]=="<job post link>"):
        posting_path = ""
    else:
        posting_path = args["job post link"]

    return evaluate_resume(about_me=about_me, resume_file=resume_file, posting_path=posting_path)







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
    if ("about me" not in args or args["about me"] == "" or args["about me"]=="<about me>"):
        about_me = ""
    else:
        about_me = args["about me"]
    if ("job post link" not in args or args["job post link"]=="" or args["job post link"]=="<job post link>"):
        posting_path = ""
    else:
        posting_path = args["job post link"]
        
    return evaluate_resume(about_me=about_me, resume_file=resume_file, posting_path=posting_path)


def create_resume_evaluator_tool() -> List[Tool]:

    """ Input parser: input is user's input as a string of text. This function takes in text and parses it into JSON format. 
    
    Then it calls the processing_resume function to process the JSON data. """

    name = "resume_evaluator"
    parameters = '{{"about me":"<about me>", "resume file":"<resume file>", "job post link":"<job post link>"}}' 
    output = '{{"file_path":"<file_path>"}}'
    description = f"""Helps to evaluate a resume. Use this tool more than any other tool when user asks to evaluate, review, help with a resume. 
    Input should be a single string strictly in the following JSON format: {parameters} \n
     Leave value blank if there's no information provided. DO NOT MAKE STUFF UP. 
     (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else) \n
     Output should be using the "get download link" tool in the following single string JSON format: {output}
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
    # my_job_title = 'Data Analyst'
    # my_resume_file = './resume_samples/resume2023v3.txt'
    # job_posting = "./uploads/file/data_analyst_SC.txt"
    # company = "Southern Company"
    # evaluate_resume(my_job_title =my_job_title, company = company, resume_file=my_resume_file, posting_path = job_posting)
    my_resume_file = "./resume_samples/college-student-resume-example.txt"
    evaluate_resume(resume_file=my_resume_file)



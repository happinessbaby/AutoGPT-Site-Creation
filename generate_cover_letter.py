# Import the necessary modules
import os
from openai_api import get_completion
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from basic_utils import read_txt, write_to_docx
from common_utils import (extract_personal_information, get_web_resources,  retrieve_from_db,extract_education_level, extract_work_experience_level, search_relevancy_advice,
                         extract_posting_information, create_sample_tools, extract_job_title, search_related_samples)
from datetime import date
from pathlib import Path
import json
from json import JSONDecodeError
from langchain.agents import AgentType, Tool, initialize_agent
from multiprocessing import Process, Queue, Value
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from typing import List
from langchain_utils import create_search_tools, create_summary_chain, generate_multifunction_response
from langchain.tools import tool
import uuid
import docx




from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
# cover_letter_path = os.environ["COVER_LETTER_PATH"]
cover_letter_samples_path = os.environ["COVER_LETTER_SAMPLES_PATH"]
# TODO: caching and serialization of llm
llm = ChatOpenAI(temperature=0.5, cache=False)
# llm = OpenAI(temperature=0, top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)
embeddings = OpenAIEmbeddings()
delimiter = "####"
delimiter1 = "****"
delimiter2 = "'''"
delimiter3 = '---'
delimiter4 = '////'
delimiter5 = '~~~~'


      
def generate_basic_cover_letter(my_job_title="", company="", resume_file="",  posting_path="") -> str:
    
    """ Main function that generates the cover letter.
    
    Keyword Args:

      my_job_title (str): job applying for

      company (str): company applying for

      resume_file (str): path to the resume file in txt format

      posting_path (str): path to job posting in txt format

    Returns:

      cover letter as a text string
     
    """
    
    dirname, fname = os.path.split(resume_file)
    res_path = os.path.join(dirname, fname+"_cl"+".txt")

    # Get resume as text format
    resume_content = read_txt(resume_file)
    # Get personal information from resume
    personal_info_dict = extract_personal_information(resume_content)
  
    
    job_specification = ""
    job_description = ""
    # Get job information from posting link, if provided
    if (Path(posting_path).is_file()):
      prompt_template = """Identity the job position, company then provide a summary of the following job posting:
        {text} \n
        Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
      """
      job_specification = create_summary_chain(posting_path, prompt_template)
      posting = read_txt(posting_path)
      posting_info_dict=extract_posting_information(posting)
      my_job_title = posting_info_dict["job"]
      company = posting_info_dict["company"]
    # Get general job description otherwise
    else:
      if (my_job_title==""):
          my_job_title = extract_job_title(resume_content)
      job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed. """
      job_description = get_web_resources(job_query) 
       
 
    # Search for a job description of the job title

    # Search for company description if company name is provided
    company_description=""
    if (company!=""):
      company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values.                       
                          Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output -1."""
      company_description = get_web_resources(company_query)


    education_level = extract_education_level(resume_content)
     # work_experience = extract_work_experience_level(resume, my_job_title)
    

    # Get resume's relevant and irrelevant info for job: few-shot learning works great here
    query_relevancy = f"""Determine the relevant and irrelevant information contained in the resume document delimited with {delimiter} characters.

      resume: {delimiter}{resume_content}{delimiter} \n

      Generate a list of irrelevant information that should not be included in the cover letter and a list of relevant information that should be included in the cover letter.
       
      Remember to use either job specification or general job description as your guideline. Don't forget to use your tool "search_relevancy_advice".

      job specification: {job_specification}

      general job description: {job_description} \n

        Your answer should be detailed and only from the resume. Please also provide your reasoning too. 
        
        For example, your answer may look like this:

        Relevant information:

        1. Experience as a Big Data Engineer: using Spark and Python are transferable skills to using Panda in data analysis

        Irrelevant information:

        1. Education in Management of Human Resources is not directly related to skills required for a data analyst 

    
        """
  
    tool = [search_relevancy_advice]
    relevancy = generate_multifunction_response(query_relevancy, tool)


    # Get adviced from web data on personalized best practices
    advice_query = f"""Best practices when writing a cover letter for {my_job_title} with the given information:
      highest level of education: {education_level}
      work experience level:   """
    advices = retrieve_from_db(advice_query)

    # Get sample comparisons
    related_samples = search_related_samples(my_job_title, cover_letter_samples_path)
    sample_tools, tool_names = create_sample_tools(related_samples, "cover_letter")
    query_samples = f""" 

    Sample cover letters are provided in your tools. Research {str(tool_names)} and answer the following question: and answer the following question:

      1. What common features these cover letters share?

      """
    shared_practices = generate_multifunction_response(query_samples, sample_tools)
    
 
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read

    # Step wise instructions: https://learn.deeplearning.ai/chatgpt-building-system/lesson/5/chain-of-thought-reasoning

    template_string2 = """You are a professional cover letter writer. A Human client has asked you to generate a cover letter for them.
  
        The content you are to use as reference to create the cover letter is delimited with {delimiter} characters.

        Always use this as a context when writing the cover letter. Do not write out of context and do not make anything up. Anything that should be included but unavailable can be put in brackets. 

        content: {delimiter}{content}{delimiter}. \n

      Step 1: You are given two lists of information delimited with characters. One is irrelevant to applying for {job} and the other is relevant. 

        Use them as a reference when determining what to include and what not to include in the cover letter. 
    
        information list: {relevancy}.  \n

      Step 2: You're given a list of best practices shared among exceptional cover letters. Use them when generating the cover letter. 

        best practices: {practices}. \n

      Step 3: You are also given some expert advices. Keep them in mind when generating the cover letter.

        expert advices: {advices}

      Step 3: You're provided with some company informtion. 

        Use it to make the cover letter cater to the company values. 

        company information: {company_description}.  \n

    
      Step 4: Change all personal information of the cover letter to the following. Do not incude them if they are -1 or empty: 

        name: {name}. \

        email: {email}. \

        phone number: {phone}. \
        
        today's date: {date}. \
        
        company they are applying to: {company}. \

        job position they are applying for: {job}. \
    
      Step 5: Generate the cover letter using what you've learned in Step 1 through Step 4. Do not make stuff up. 
    
      Use the following format:
        Step 1:{delimiter4} <step 1 reasoning>
        Step 2:{delimiter4} <step 2 reasoning>
        Step 3:{delimiter4} <step 3 reasoning>
        Step 4:{delimiter4} <step 4 reasoning>
        Step 5:{delimiter4} <the cover letter you generate>

      Make sure to include {delimiter4} to separate every step.
    """
    
    prompt_template = ChatPromptTemplate.from_template(template_string2)
    # print(prompt_template.messages[0].prompt.input_variables)

    cover_letter_message = prompt_template.format_messages(
                    name = personal_info_dict.get('name', ""),
                    email = personal_info_dict.get('email', ""),
                    phone = personal_info_dict.get('phone', ""),
                    date = date.today(),
                    company = company,
                    job = my_job_title,
                    content=resume_content,
                    relevancy=relevancy, 
                    practices = shared_practices, 
                    advices = advices,
                    company_description = company_description, 
                    delimiter = delimiter,
                    delimiter4 = delimiter4,
    )

    my_cover_letter = llm(cover_letter_message).content
    cover_letter = get_completion(f"Extract the cover letter from text: {my_cover_letter}")
    # write cover letter to file  
    doc = docx.Document()
    # with open(res_path, 'w') as f:
    #     f.write(cover_letter)
    #     print(f"Sucessfully written cover letter to {res_path}")
    write_to_docx(doc, cover_letter, "cover_letter", res_path)
    return cover_letter


@tool(return_direct=True)
def cover_letter_generator(json_request:str) -> str:
    
    """Helps to generate a cover letter. Use this tool more than any other tool when user asks to write a cover letter.  

    Input should be a single string strictly in the following JSON format:  '{{"job":"<job>", "company":"<company>", "resume file":"<resume file>", "job post link": "<job post link>"}}'\n

    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else) \n

    Output should be the exact cover letter that's generated for the user word for word. 
    """
    
    try:
      json_request = json_request.strip("'<>() ").replace('\'', '\"')
      args = json.loads(json_request)
    except JSONDecodeError as e:
      print(f"JSON DECODE ERROR: {e}")
      return "Format in a single string JSON and try again."
   
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


    return generate_basic_cover_letter(my_job_title=job, company=company, resume_file=resume_file, posting_path=posting_path)




# def processing_cover_letter(json_request: str) -> None:
    
#     """ Input parser: input is LLM's action_input in JSON format. This function then processes the JSON data and feeds them to the cover letter generator. """

#     try:
#         args = json.loads(json_request)
#     except JSONDecodeError:
#         return "Format in JSON and try again." 
#     # if resume doesn't exist, ask for resume
#     if ("resume file" not in args or args["resume file"]=="" or args["resume file"]=="<resume file>"):
#       return "Can you provide your resume so I can further assist you? "
#     else:
#       # may need to clean up the path first
#         resume_file = args["resume file"]
#     if ("job" not in args or args["job"] == "" or args["job"]=="<job>"):
#         job = ""
#     else:
#       job = args["job"]
#     if ("company" not in args or args["company"] == "" or args["company"]=="<company>"):
#         company = ""
#     else:
#         company = args["company"]
#     if ("job post link" not in args or args["job post link"]=="" or args["job post link"]=="<job post link>"):
#         posting_path = ""
#     else:
#         posting_path = args["job post link"]


#     return generate_basic_cover_letter(my_job_title=job, company=company, resume_file=resume_file, posting_path=posting_path)


    
# def create_cover_letter_generator_tool() -> List[Tool]:
    
#     """ Input parser: input is user's input as a string of text. This function takes in text and parses it into JSON format. 
    
#     Then it calls the processing_cover_letter function to process the JSON data. """
    
#     name = "cover_letter_generator"
#     parameters = '{{"job":"<job>", "company":"<company>", "resume file":"<resume file>", "job post link": "<job post link>"}}'
#     description = f"""Helps to generate a cover letter. Use this tool more than any other tool when user asks to write a cover letter.  
#     Input should be JSON in the following format: {parameters} \n
#     (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else) 
#     """
#     tools = [
#         Tool(
#         name = name,
#         func =processing_cover_letter,
#         description = description, 
#         verbose = False,
#         )
#     ]
#     print("Sucessfully created cover letter generator tool. ")
#     return tools

  
    
 
if __name__ == '__main__':
    # test run defaults, change for yours (only resume_file cannot be left empty)
    my_job_title = 'data analyst'
    my_resume_file = 'resume_samples/resume2023v2.txt'
    generate_basic_cover_letter(resume_file = my_resume_file)




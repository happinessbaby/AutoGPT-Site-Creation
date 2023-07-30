import os
from openai_api import evaluate_content, check_content_safety, get_completion
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate
from basic_utils import read_txt
from common_utils import compare_samples, get_web_resources, retrieve_from_db, get_job_relevancy, extract_posting_information, get_summary
from pathlib import Path


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# TBD: caching and serialization of llm
llm = ChatOpenAI(temperature=0.0, cache=False)
# llm = OpenAI(temperature=0, top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)
embeddings = OpenAIEmbeddings()
# randomize delimiters from a delimiters list to prevent prompt injection
delimiter = "####"
delimiter1 = "````"
delimiter2 = "////"
delimiter3 = "<<<<"
delimiter4 = "****"

my_job_title = 'accountant'
my_resume_file = 'resume_samples/sample1.txt'
resume_advice_path = './web_data/resume/'
resume_samples_path = './resume_samples/'
posting_path = "./uploads/posting/accountant.txt"


    
def evaluate_resume(my_job_title, company="", read_path = my_resume_file, res_path="./static/advice/advice_rewrite.txt", posting_path=""):

    resume = read_txt(read_path)

    resume_fields = extract_fields(resume)

    resume_fields = resume_fields.split(",")
    print(f"{resume_fields}")

    job_specification = ""
    if (Path(posting_path).is_file()):
      job_specification = get_summary(posting_path)
      posting = read_txt(posting_path)
      posting_info_dict=extract_posting_information(posting)
      my_job_title = posting_info_dict["job"]
      company = posting_info_dict["company"]
 
    query_job  = f"""Research what a {my_job_title} does, including details of the common skills, responsibilities, education, experience needed for the job."""
    job_description = get_web_resources(query_job, "google")

    company_description=""
    if (company!=""):
      company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values.
                          
                          Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output you don't know"""
      
      company_description = get_web_resources(company_query, "wiki")


    for field in resume_fields:

      print(f"CURRENT FIELD IS: {field}")

      field_content = get_field_content(resume, field)
        
      
      query_relevancy = f"""Determine the relevant and irrelevant information contained in the resume field.

        You are  provided with job specification for an opening position. 
        
        Use it as a primarily guidelines when generating your answer. 

        You are also provided with a general job decription of the requirement of {my_job_title}. 
        
        Use it as a secondary guideline when forming your answer.

        If job specification is not provided, use general job description as your primarily guideline. 


        reusme field: {field_content}\n

        job specification: {job_specification}\n

        general job description: {job_description} \n


        Generate a list of irrelevant information that should not be included in the resume and a list of relevant information that should be included in the field. 

          """
      relevancy = get_job_relevancy(read_path, query_relevancy)

      query_advice =  f"how to best wriite {field} for resume?"

      resume_advices = retrieve_from_db(resume_advice_path, query_advice)

      query_samples = f""" 
        Research sample resume provided. 

        If the resume contains a field that's related to {field}, answer the following question. Otherwise, ignore the questions: 

        1. common noun keywords 

        2. common action keywords

        """
      # practices = compare_samples(my_job_title,  query_samples, resume_samples_path, "resume")


      template_string = """" Your task is to analyze and help improve the content of resume field {field}. 

      The content of the field is delimiter with {delimiter} characters. Always use this as contenxt and do not make things up. 

      field content: {delimiter}{field_content}{delimiter}
      

      Step 1: You're given some expert advices on how to write {field} . Keep these advices in mind for the next steps.

          expert advices: {delimiter1}{advices}{delimiter1}  \n


      step 2: You are given two lists of information delimited with {delimiter2} characters. One is content to be included in the {field} and the other is content to be removed. 

          Use them as to generate your answer. 

          information list: {delimiter2}{relevancy}{delimiter2} \n

      Step 3: You're provided with some company informtion and job specification

        Use it to make the resume field {field} cater to the company and job specification more. 

        If company information and/or job specifcation do not pertain to the resume field, skip this step. 

        company information: {company_description}.  \n

        job specification: {job_specification}.    \n

      Step 4: Based on what you gathered in Step 1 through 3, rewrite the resume field {field}. Do not make up things. 

      Use the following format:
          Step 1:{delimiter4} <step 1 reasoning>
          Step 2:{delimiter4} <step 2 reasoning>
          Step 3:{delimiter4} <step 3 reasoning>
          Step 4:{delimiter4} <rewrite the resume field>


        Make sure to include {delimiter4} to separate every step.
      
      """

      prompt_template = ChatPromptTemplate.from_template(template_string)
      upgrade_resume_message = prompt_template.format_messages(
          field = field,
          field_content = field_content,
          job = my_job_title,
          advices=resume_advices,
          relevancy = relevancy,
          company_description = company_description,
          job_specification = job_specification, 
          delimiter = delimiter,
          delimiter1 = delimiter1, 
          delimiter2 = delimiter2,
          delimiter4 = delimiter4, 
          
      )
      my_advice = llm(upgrade_resume_message).content

      # Check potential harmful content in response
      if (check_content_safety(text_str=my_advice)):   
          # Write the cover letter to a file
          with open(res_path, 'a') as f:
              try:
                  f.write(my_advice)
                  print("ALL SUCCESS")
              except Exception as e:
                  print("FAILED")
                  # Error logging




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

def get_field_content(resume, field):
   
   
   query = f"""Retrieve all the content of field {field} from the resume file delimiter with {delimiter} charactres.

      resume: {delimiter}{resume}{delimiter}
    """

   response = get_completion(query)
   print(response)
   return response



if __name__ == '__main__':
    evaluate_resume(my_job_title)
 
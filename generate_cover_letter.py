# Import the necessary modules
import os
from openai_api import get_completion, evaluate_content, check_content_safety
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from basic_utils import read_txt
from common_utils import extract_personal_information, get_web_resources,  get_job_relevancy, retrieve_from_db, get_summary, extract_posting_information, compare_samples
from samples import cover_letter_samples_dict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# TBD: caching and serialization of llm
llm = ChatOpenAI(temperature=0.0, cache=False)
# llm = OpenAI(temperature=0, top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)
embeddings = OpenAIEmbeddings()
delimiter = "####"
delimiter1 = "****"
delimiter2 = "'''"
delimiter3 = '---'
delimiter4 = '////'
delimiter5 = '~~~~'

# test run defaults, change for yours
my_job_title = 'accountant'
my_resume_file = 'resume_samples/sample1.txt'
cover_letter_advice_path = "./web_data/cover_letter/"
posting_path = "./uploads/posting/accountant.txt"
cover_letter_samples_path = "./sample_cover_letters/"



def generate_basic_cover_letter(my_job_title, company="", read_path=my_resume_file, res_path= "./static/cover_letter/cover_letter.txt", posting_path=""):
    
    resume_content = read_txt(read_path)
    # Get personal information from resume
    personal_info_dict = extract_personal_information(resume_content)

    job_specification = ""
    if (Path(posting_path).is_file()):
      job_specification = get_summary(posting_path)
      posting = read_txt(posting_path)
      posting_info_dict=extract_posting_information(posting)
      my_job_title = posting_info_dict["job"]
      company = posting_info_dict["company"]

    # Get advices on cover letter
    advice_query = "What are some best practices when writing a cover letter?"
    # advices = retrieve_from_vectorstore(embeddings, advice_query, index_name="redis_cover_letter_advice")
    advices = retrieve_from_db(cover_letter_advice_path, advice_query)
  
    # Get job description using Google serach
    job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed. """
    job_description = get_web_resources(job_query, "google")

    # Get company descriptiong using Wikipedia lookup
    # for future probably will need to go into company site and scrape more information
    company_description=""
    if (company!=""):
      company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values.
                          
                          Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output you don't know"""
      company_description = get_web_resources(company_query, "wiki")


    query_relevancy = f"""Determine the relevant and irrelevant information contained in the resume document delimited with {delimiter} characters.

      You are  provided with job specification for an opening position, delimiter with {delimiter2} characters. They are delimited with {delimiter2} characters. 
      
      Use it as a primarily guidelines when generating your answer. 

      You are also provided with a general job decription of the requirement of {my_job_title}, which is delimited with {delimiter1} charactres. 
      
      Use it as a secondary guideline when forming your answer.

      If job specification is not provided, use general job description as your primarily guideline. 


      resume document: {delimiter}{resume_content}{delimiter} \n

      job specification: {delimiter2}{job_specification}{delimiter2} \n

      general job description: {delimiter1}{job_description}{delimiter1} \n


      Generate a list of irrelevant information that should not be included in the cover letter and a list of relevant information that should be included in the cover letter. 

        """
    relevancy = get_job_relevancy(read_path, query_relevancy)
    # Get cover letter examples
    query_samples = f""" 
      Step 3: Research sample cover letters provided. You're also given some expert advice on some common practices when writing a cover letter.

      expert advices: {advices} 

      Reference the samples and use the expert advices to answer the following questions: 

      1. word count range

      2. style and intonation

      3. common keywords, be specific

      4. three most important key content

      """
    practices = compare_samples(my_job_title,  query_samples, cover_letter_samples_path)


    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read

    # Try using a step wise instruction as seen in: https://learn.deeplearning.ai/chatgpt-building-system/lesson/5/chain-of-thought-reasoning

    template_string2 = """Generate a cover letter for a person applying for {job} at {company} using the following information. 

        The content you are to use as reference to create the cover letter is delimited with {delimiter} characters.

        Always use this as a context when writing the cover letter. Do not write out of context and do not make anything up. 

        content: {delimiter}{content}{delimiter}. \n

      Step 1: You are given two lists of information delimited with {delimiter2} characters. One is irrelevant to applying for {job} and the other is relevant. 

        Use them as a reference when determining what to include and what not to include in the cover letter. 
     
        information list: {delimiter2}{relevancy}{delimiter2}.  \n

      Step 2: You're given a list of best practices when writing the cover letter. It is delimited with {delimiter1} characters.
      
        Use it as a guideline when generating the cover letter.

        best practices: {delimiter1}{practices}{delimiter1}. \n

      Step 3: You're provided with some company informtion delimited by {delimiter3} characters. 

        Use it to make the cover letter cater to the company. 

        company information: {delimiter3}{company_description}{delimiter3}.  \n

    
      Step 4: Change all personal information of the cover letter to the following. Do not incude them if they are -1 or empty: 

        name: {name}. \

        email: {email}. \

        phone number: {phone}. \
        
        today's date: {date}. \
        
        company they are applying to: {company}. \

        job position they are applying for: {job}. \
    
      Step 5: Generate the cover letter.
    
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
                    name = personal_info_dict.get('name'),
                    email = personal_info_dict.get('email'),
                    phone = personal_info_dict.get('phone'),
                    date = date.today(),
                    company = company,
                    job = my_job_title,
                    content=resume_content,
                    relevancy=relevancy, 
                    practices = practices, 
                    company_description = company_description, 
                    delimiter = delimiter,
                    delimiter1 = delimiter1, 
                    delimiter2 = delimiter2,
                    delimiter3 = delimiter3,
                    delimiter4 = delimiter4,
    )

    my_cover_letter = llm(cover_letter_message).content

    # my_cover_letter= my_cover_letter.split(delimiter4)[-1].strip()

    # Check potential harmful content in response
    if (check_content_safety(text_str=my_cover_letter)):   
        # Validate cover letter
        if (evaluate_content(my_cover_letter, "cover letter")):
            # Write the cover letter to a file
            with open(res_path, 'w') as f:
                try:
                    f.write(my_cover_letter)
                    print("ALL SUCCESS")
                    return True
                except Exception as e:
                    print("FAILED")
                    return False
                    # Error logging
    
        

# Call the function to generate the cover letter
 
if __name__ == '__main__':
    generate_basic_cover_letter(my_job_title)



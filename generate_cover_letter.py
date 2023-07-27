# Import the necessary modules
import os
from openai_api import get_completion, evaluate_response, check_content_safety
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from basic_utils import read_txt
from common_utils import extract_personal_information, get_web_resources, fetch_similar_samples, retrieve_from_vectorstore, get_job_relevancy
from samples import cover_letter_samples_dict
from datetime import date

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



def generate_basic_cover_letter(my_job_title, company="openai", read_path=my_resume_file, res_path= "./static/cover_letter/cover_letter.txt"):
    
    resume_content = read_txt(read_path)
    # Get personal information from resume
    personal_info_dict = extract_personal_information(resume_content)

    # Get advices on cover letter
    advice_query = "What to include and what not to include in a cover letter?"
    advices = retrieve_from_vectorstore(embeddings, advice_query, index_name="redis_cl_test")
  
    # Get job description using Google serach
    job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed. """
    job_description = get_web_resources(job_query, "google")

    # Get company descriptiong using Wikipedia lookup
    company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values, and the products or services that it offers.
                        
                        Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output you don't know"""
    company_description = get_web_resources(company_query, "wiki")


    query_relevancy = f"""Determine the relevant and irrelevant information contained in the resume document delimited with {delimiter} characters.

      You are provided with a decription of the job requirement of {my_job_title}, which is delimited with {delimiter1} charactres. Use it as a guideline when forming your answer.

      You are also provided with some expert advices are what to include and what not to include in a cover letter. They are delimited with {delimiter2} characters. Use them as a guideline too. 

      resume document: {delimiter}{resume_content}{delimiter} \n

      job requirement of {my_job_title}: {delimiter1}{job_description}{delimiter1} \n

      expert advices: {delimiter2}{advices}{delimiter2} \n

      Generate a list of irrelevant information that should not be included in the cover letter and a list of relevant information that should be included in the cover letter. 

        """
    relevancy = get_job_relevancy(read_path, query_relevancy)
    # Get cover letter examples
    query_samples = f""" 
      Step 3: Research sample cover letters provided.

        You're also give two list of information delimiter with {delimiter1} characters. One is irrelevant and should not be included in the cover letter, The other list is relevant.

        Based on what you know about how sample cover letters are written, fine-tune the two lists.

        relevant and irrelevant lists: {delimiter1}{relevancy}{delimiter1} \n
        
      """
    comparison = fetch_similar_samples(embeddings, my_job_title, cover_letter_samples_dict, query_samples)
    if (comparison==""):
        print("no cover letter samples for comparison")
        comparison = relevancy


    

    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read

    # Try using a step wise instruction as seen in: https://learn.deeplearning.ai/chatgpt-building-system/lesson/5/chain-of-thought-reasoning

    template_string2 = """Generate a cover letter for a person applying for {job} at {company} using the following information. 

      Step1: The content you are to use as reference to create the cover letter is delimited with {delimiter} characters.

        Always use this as a context when writing the cover letter. Do not write out of context and do not make anything up. 

        content: {delimiter}{content}{delimiter}. \n

      Step 2: You are given two lists of information delimited with {delimiter1} characters. One is irrelevant to applying for {job} and the other is relevant. 

        Use them as a reference when determining what to include and what to not include in the cover letter. 
     
        information list: {delimiter2}{comparison}{delimiter2}.  \n

      Step 3: You're provided with some company informtion delimited by {delimiter3} characters. Use it to make the cover letter more specific to the company.

        company information: {delimiter3}{company_description}{delimiter3} \n
    
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
                    comparison = comparison, 
                    company_description = company_description,
                    delimiter = delimiter,
                    delimiter1 = delimiter1, 
                    delimiter2 = delimiter2,
                    delimiter3 = delimiter3,
                    delimiter4 = delimiter4,
    )
    # FOR THE FUTURE, THIS LLM HERE CAN BE CUSTOMLY TRAINED
    my_cover_letter = llm(cover_letter_message).content

    my_cover_letter= my_cover_letter.split(delimiter4)[-1].strip()

    # Check potential harmful content in response
    if (check_content_safety(text_str=my_cover_letter)):   
        # Validate cover letter
        if (evaluate_response(my_cover_letter)):
            # Write the cover letter to a file
            with open(res_path, 'w') as f:
                try:
                    f.write(my_cover_letter)
                    print("ALL SUCCESS")
                except Exception as e:
                    print("FAILED")
                    # Error logging
    
        

# Call the function to generate the cover letter
 
if __name__ == '__main__':
    generate_basic_cover_letter(my_job_title)



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



def generate_basic_cover_letter(my_job_title, company="abc", read_path=my_resume_file, res_path= "./static/cover_letter/cover_letter.txt"):
    
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


    query_irrelevancy = f"""Determine all the irrelevant information contained in the resume document delimited with {delimiter} characters that are unrelated to {my_job_title} 

      You are provided with a decription of the job requirement of {my_job_title}, which is delimited with {delimiter1} charactres. Use it as a guideline when forming your answer.

      You are also provided with some expert advices are what to include and what not to include in a cover letter. They are delimited with {delimiter2} characters. Use them as a guideline too. 

      resume document: {delimiter}{resume_content}{delimiter} \n

      job requirement of {my_job_title}: {delimiter1}{job_description}{delimiter1} \n

      expert advices: {delimiter2}{advices}{delimiter2} \n

      Generate a list of irrelevant information that should not be included in the cover letter.

        """
    irrelevancy = get_job_relevancy(read_path, query_irrelevancy)
    # Get cover letter examples
    cover_letter_examples = fetch_similar_samples(embeddings, my_job_title, cover_letter_samples_dict)

    query_relevancy = f""" 
      Step 3: Research example cover letters provided. Each example is delimited with {delimiter3} characters.

        From these examples, summarize the information that is useful for generating a cover letter.

        You're also given expert advices, which is delimited with {delimiter} characters, on what to include in a cover letter. 

       Generate a list of content that should be included in a cover letter.
       
       You should be base your judgment on the cover letter examples and the expert advices. 

       expert advices: {delimiter}{advices}{delimiter} \n
        
        example: {cover_letter_examples}. \n

      """
    relevancy = get_job_relevancy(read_path, query_relevancy)
    


    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read

    # Try using a step wise instruction as seen in: https://learn.deeplearning.ai/chatgpt-building-system/lesson/5/chain-of-thought-reasoning

    template_string2 = """Generate a cover letter for a person applying for {job} at {company} using the following information. 

      Step1: The content you are to use as reference to create the cover letter is delimited with {delimiter} characters.

        Always use this use a point of reference when asked what to include or what not to include in the cover letter. 

        content: {delimiter}{content}{delimiter}. \n

      Step 2: You are given a list of irrelevancy information delimited with {delimiter2} characters that is considered irrelevant to applying for {job}. 
    
        They should not be included in the cover letter. 
     
        irrelevancy information: {delimiter2}{relevancy}{delimiter2}.  \n

      Step 3: You are given a list of relevancy information delimiter with {delimiter3} characters that should be included in the cover letter.

        If they exist in the content, they should be included in the cover letter.

        relevancy information: {delimiter3}{irrelevancy}{delimiter3}. \n

      Step 4: You're provided with some company informtion delimited by {delimiter5} characters. Use it to make the cover letter more specific to the company.

        company information: {delimiter5}{company_description}{delimiter5} \n
    
      Step 5: Change all personal information of the cover letter to the following. Do not incude them if they are -1 or empty: 

        name: {name}. \

        email: {email}. \

        phone number: {phone}. \
        
        today's date: {date}. \
        
        company they are applying to: {company}. \

        job position they are applying for: {job}. \
    
      Step 6: Generate the cover letter.
    
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
                    irrelevancy = irrelevancy,
                    relevancy = relevancy, 
                    company_description = company_description,
                    delimiter = delimiter,
                    delimiter1 = delimiter1, 
                    delimiter2 = delimiter2,
                    delimiter3 = delimiter3,
                    delimiter4 = delimiter4,
                    delimiter5 = delimiter5,
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



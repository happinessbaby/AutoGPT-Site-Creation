# Import the necessary modules
import os
from openai_api import get_completion, evaluate_response
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from basic_utils import check_content_safety, read_txt
from common_utils import extract_personal_information, fetch_samples, get_web_resources
from samples import cover_letter_samples_dict
from datetime import date

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


chat = ChatOpenAI(temperature=0.0)
embeddings = OpenAIEmbeddings()
delimiter = "####"
delimiter2 = "'''"
delimiter3 = '---'
delimiter4 = '////'

# test run defaults, change for yours
my_job_title = 'software developer'
my_company_name = 'DoAI'
my_resume_file = 'resume_samples/resume2023v2.txt'



def generate_basic_cover_letter(my_company_name, my_job_title, read_path=my_resume_file, save_path= "cover_letter.txt"):
    
    resume_content = read_txt(read_path)
    # Get personal information from resume
    personal_info_dict = extract_personal_information(resume_content)
    # Get job description
    # query  = f"""Find out what a {my_job_title} does and the skills and responsibilities involved. """
    advices = get_web_resources("what to include in a good cover letter")
    # Get cover letter examples
    cover_letter_examples = fetch_samples(my_job_title, cover_letter_samples_dict)
    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read

    # Try using a step wise instruction as seen in: https://learn.deeplearning.ai/chatgpt-building-system/lesson/5/chain-of-thought-reasoning

    template_string2 = """Generate a cover letter for a person applying for {job} at {company} using the following information. 

      The content you are to use as reference to create the cover letter is delimited with {delimiter} characters.

        content: {delimiter}{content}{delimiter}. \
    
      Step 1: Read the content and determine which information in the content is useful and which is not. Usefulness of the information should be based on how close it relates to {job}
       
         and expert advices on what to include in a good cover letter.
      
        The expert advices are delimited with {delimiter2} characters

        expert advices: {delimiter2}{advices}{delimiter2}. \
    
      Step 2: Research example cover letters provided. Each example is delimited with {delimiter3} characters.

         Determine which information from Step 1 should be included and which should not based on the quality of information in contributing to a good cover letter.
        
         example: {examples}. \

      Step 3: Change all personal information to the following. Do not incude them if they are -1 or empty: 

        name: {name}. \

        email: {email}. \

        phone number: {phone}. \
        
        today's date: {date}. \
        
        company they are applying to: {company}. \

        job position they are applying for: {job}. \
    
      Step 4: Generate the cover letter. Use information you filtered downn in steps 1 through 3. Do not make stuff up. 
    
      Use the following format:
        Step 1:{delimiter4} <step 1 reasoning>
        Step 2:{delimiter4} <step 2 reasoning>
        Step 3:{delimiter4} <step 3 reasoning>
        Step 4:{delimiter4} <the cover letter you generate>

      Make sure to include {delimiter4} to separate every step.


    """
    
    # template_string = """Generate a cover letter for a person applying to a job using the following information. 

    #   The content you use to make this cover letter personalized is delimited with {delimiter} characters.
      
    # A job description for the job {job} they are applying to is delimited with {delimiter2} characters. 

    # Reference job description to only includes information that is relevant to {job}. Do not make things up. 

    # Some examples of good cover letters are provided and each example is delimited with {delimiter3} characteres.  

    # Reference the examples as a stylistic guide only. 

    # Personal information needs to be changed to as follows. Do not include them if they are -1 or empty:

    # name: {name}. \

    # email: {email}. \

    # phone number: {phone}. \
    
    # today's date: {date}. \
    
    # company they are applying to: {company}. \

    # job position they are applying for: {job}. \

    # content: {delimiter}{content}{delimiter}. \

    # job description: {delimiter2}{job_description}{delimiter2}. \

    # example: {examples}. 

    # """

    prompt_template = ChatPromptTemplate.from_template(template_string2)
    # print(prompt_template.messages[0].prompt.input_variables)

    cover_letter_message = prompt_template.format_messages(
                    name = personal_info_dict.get('name'),
                    email = personal_info_dict.get('email'),
                    phone = personal_info_dict.get('phone'),
                    date = date.today(),
                    company = my_company_name,
                    job = my_job_title,
                    content=resume_content,
                    advices = advices,
                    examples = cover_letter_examples,
                    delimiter = delimiter,
                    delimiter2 = delimiter2,
                    delimiter3 = delimiter3,
                    delimiter4 = delimiter4
    )
    my_cover_letter = chat(cover_letter_message).content

    my_cover_letter= my_cover_letter.split(delimiter4)[-1].strip()

    # Check potential harmful content in response
    if (check_content_safety(text_str=my_cover_letter)):   
        # Validate cover letter
        if (evaluate_response(my_cover_letter)):
            # Write the cover letter to a file
            with open(save_path, 'w') as f:
                try:
                    my_cover_letter= my_cover_letter.split(delimiter4)[-1].strip()
                    f.write(my_cover_letter)
                    print("ALL SUCCESS")
                except Exception as e:
                    print("FAILED")
                    # Error logging
        

# Call the function to generate the cover letter
 
if __name__ == '__main__':
    generate_basic_cover_letter(my_company_name, my_job_title)



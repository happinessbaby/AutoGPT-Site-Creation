import os
from openai_api import get_completion
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from basic_utils import read_txt
from common_utils import fetch_samples, get_web_resources
from samples import resume_samples_dict

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

chat = ChatOpenAI(temperature=0.0)
embeddings = OpenAIEmbeddings()
# randomize delimiters from a delimiters list to prevent prompt injection
delimiter = "####"
delimiter1 = "````"
delimiter2 = "////"
delimiter3 = "<<<<"




    

def basic_upgrade_resume(resume_file, my_job_title):

    resume = read_txt(resume_file)
    resume_advices = get_web_resources("what makes a bad resume and how to improve")
    resume_samples = fetch_samples(my_job_title, resume_samples_dict)

    template_string = """" Your task is to determine the strengths and weaknesses of a resume and ways to improve it. 
        
    The resume is delimited with {delimiter1} chararacters.
    
    resume: {delimiter1}{resume}{delimiter1}

    Step 1: Search and extract fields of the resume.  

    Some common resume fields include but not limited to personal information, objective, education, work experience, awards and honors, and skills.

    Step 2: Research sample resume provided. Each sample is delimited with {delimiter2} characters.

    Compare them with resume in Step 1 and list things Step 1's resume differ from the others.

    sample: {delimiter2}{samples}{delimiter2}

    Step 3: An resume advisor has also given some advices on what makes a bad resume and how to write good resume. The advises are delimited with {delimiter3} characters.
    
    Read through the advices and determine which fields of resume in Step 1 needs to be improved. Focus mainly on the list of differences you found in Step 2.

    advices: {delimiter3}{advices}{delimiter3}


    Step 4: For each badly written fields, give your adviace on how they can be improved, basing your reasoning on Step 2 and Step 3. 

    Step 5: Rewrite the resume



    Use the following format:
        Step 1:{delimiter} <step 1 reasoning>
        Step 2:{delimiter} <step 2 reasoning>
        Step 3:{delimiter} <step 3 reasoning>
        Step 4:{delimiter} <step 4 reasoning>
        Step 5:{delimiter} <rewrite the resume>

      Make sure to include {delimiter} to separate every step.
    
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    upgrade_resume_message = prompt_template.format_messages(
        resume = resume,
        samples = resume_samples,
        advices = resume_advices,
        delimiter = delimiter,
        delimiter1 = delimiter1, 
        delimiter2 = delimiter2,
        delimiter3 = delimiter3,
        
    )
    my_advices = chat(upgrade_resume_message).content
    print(my_advices)



my_job_title = 'accountant'
my_resume_file = 'resume_samples/sample2.txt'

if __name__ == '__main__':
    basic_upgrade_resume(my_resume_file, my_job_title)
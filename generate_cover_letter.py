# Import the necessary modules
import os
# from openai_api import get_completion1, get_completion2
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from pathlib import Path
import markdown


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
# Define the function to generate the cover letter
chat = ChatOpenAI(temperature=0.0)

def extract_personal_information(resume_file):

    with open(resume_file, 'r') as f:
        resume = f.read()

    name_schema = ResponseSchema(name="name",
                             description="Extract the full name of the applicant. If this information is not found, output -1")
    email_schema = ResponseSchema(name="email",
                                        description="Extract the email address of the applicant. If this information is not found, output -1")
    phone_schema = ResponseSchema(name="phone",
                                        description="Extract the phone number of the applicant. If this information is not found, output -1")
    address_schema = ResponseSchema(name="address",
                                        description="Extract the home address of the applicant. If this information is not found, output -1")

    response_schemas = [name_schema, 
                        email_schema,
                        phone_schema, 
                        address_schema]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    template_string = """For the following text, extract the following information:

    name: Extract the full name of the applicant. If this information is not found, output -1\

    email: Extract the email address of the applicant. If this information is not found, output -1\

    phone number: Extract the phone number of the applicant. If this information is not found, output -1\
    
    address: Extract the home address of the applicant. If this information is not found, output -1\

    text: {text}

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=template_string)
    messages = prompt.format_messages(text=resume, 
                                format_instructions=format_instructions)
    
    response = chat(messages)
    output_dict = output_parser.parse(response.content)
    print(output_dict.get('email'))



def extract_other_information(resume_file):
    loader = TextLoader(resume_file, encoding='utf8')
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])
    query = "list all the fields in the resume in a table in markdown and summarize each other"
    response = index.query(query)
    print(response)




def generate_basic_cover_letter(my_name,  my_company_name, my_job_title, my_resume_file):
    # Read the resume file
    resume_filename = my_resume_file.filename
    with open(resume_filename, 'r') as f:
        resume = f.read()
    # print(resume)
    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read
    
    # style_string = f"Dear Hiring Manager, I am writing to express my interest in the {company_name} position at your company. My name is {my_name} ......"

    template_string = """Generate a cover letter for person with name {name} applying to company {company} for the job title {job}. 

      The content you use to generate this personalized cover letter is delimited by triple backticks. content: '''{content}'''"""

    prompt_template = ChatPromptTemplate.from_template(template_string)

    # print(prompt_template.messages[0].prompt.input_variables)

    cover_letter_message = prompt_template.format_messages(
                    # style=cover_letter_template,
                    name = my_name,
                    company = my_company_name,
                    job = my_job_title,
                    content=resume)
    my_cover_letter = chat(cover_letter_message).content
    # cover_letter = get_completion2(template_string)
    
    # Write the cover letter to a file
    file_name=Path(resume_filename).stem
    with open(f'cover_letter_{file_name}.txt', 'w') as f:
        f.write(my_cover_letter)

    return file_name




def fine_tune_cover_letter(resume_file):
    # Read the cover letter
    file_name=Path(resume_file).stem
    with open(f'cover_letter_{file_name}.txt', 'r') as f:
        resume = f.read()
    # print(resume)
    
    # fine-tune based on user needs


# Call the function to generate the cover letter
# my_name  = 'Yueqi Peng'
# my_job_title = 'AI developer'
# my_company_name = 'Google'
# my_resume_file = 'resume2023v2.txt'
# extract_personal_information(my_resume_file)
# extract_other_information(my_resume_file)
# generate_basic_cover_letter(my_name, my_company_name, my_job_title, my_resume_file)


# Import the necessary modules
import os
import markdown
# from openai_api import get_completion1, get_completion2
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.agents import load_tools, initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType
from pathlib import Path
from basic_utils import check_file_type
from langchain_utils import get_index, create_tools


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
# Define the function to generate the cover letter
chat = ChatOpenAI(temperature=0.0)
delimiter = "####"
delimiter2 = "'''"

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
    template_string = """For the following text, delimited with {delimiter} chracters, extract the following information:

    name: Extract the full name of the applicant. If this information is not found, output -1\

    email: Extract the email address of the applicant. If this information is not found, output -1\

    phone number: Extract the phone number of the applicant. If this information is not found, output -1\
    
    address: Extract the home address of the applicant. If this information is not found, output -1\

    text: {delimiter}{text}{delimiter}

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=template_string)
    messages = prompt.format_messages(text=resume, 
                                format_instructions=format_instructions,
                                delimiter=delimiter)
    
    response = chat(messages)
    personal_info_dict = output_parser.parse(response.content)
    print(personal_info_dict.get('name'))
    return personal_info_dict



def extract_resume_fields(resume_file, job_title):

    index = get_index(resume_file)

    query = f""""Your task is to summarize this text document containing a person's resume.

    List all the fields in a markdown table and summarize each field.
    
    """

    response = index.query(query)
    print(response)
    return response

def get_cover_letter_examples():
    # retrieve cover letter examples
    ### TBD
    return ""


def get_job_resources(job_title):

    tools = create_tools()
    agent= initialize_agent(
        tools, 
        chat, 
        agent=AgentType.REACT_DOCSTORE,
        handle_parsing_errors=True,
        verbose = True,
        )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent.agent,
        tools = tools,
        verbose = True,
    )
    query  = f"""Find out what a {job_title} does using Search and Lookup"""
    # response = agent_executor.run(query)
    # return response
    # TEMPORARY FIX for raise OutputParserException(f"Could not parse LLM Output: {text}") 
    try:
        response = agent_executor.run(query)
        return response
    except Exception as e:
        response = str(e)
        # if not response.startswith("Could not parse LLM output: `"):
        #     raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        # print(f"RESPONSE: {response}")
        return response





def generate_basic_cover_letter(my_company_name, my_job_title, my_resume_file):
    # Read the resume file
    
    try:
        file= my_resume_file.filename
        filename = Path(file).stem
        resume_file = file
    except:
        filename=Path(my_resume_file).stem
        resume_file = my_resume_file

    personal_info_dict = extract_personal_information(resume_file)
    resume_content = extract_resume_fields(resume_file, my_job_title)
    job_description = get_job_resources(my_job_title)
    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read
    
    # style_string = f"Dear Hiring Manager, I am writing to express my interest in the {company_name} position at your company. My name is {my_name} ......"
    template_string = """Generate a cover letter for person with name {name} applying to company {company} for the job title {job}. 

      The content you use to generate this personalized cover letter is delimited with {delimiter} characters. content: {delimiter}{content}{delimiter}.
      
    The job description you use as a reference is delimited with {delimiter2} characters. job description: {delimiter2}{job_description}{delimiter2}. """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    # print(prompt_template.messages[0].prompt.input_variables)

    cover_letter_message = prompt_template.format_messages(
                    # style=cover_letter_template,
                    name = personal_info_dict.get('name'),
                    company = my_company_name,
                    job = my_job_title,
                    content=resume_content,
                    job_description = job_description,
                    delimiter = delimiter,
                    delimiter2 = delimiter2)
    my_cover_letter = chat(cover_letter_message).content
    # cover_letter = get_completion2(template_string)
    
    # Write the cover letter to a file
    with open(f'cover_letter_{filename}.txt', 'w') as f:
        f.write(my_cover_letter)

    return filename



# Call the function to generate the cover letter

my_job_title = 'MLOps engineer'
my_company_name = 'Facebook'
my_resume_file = 'resume2023v2.txt'
# extract_personal_information(my_resume_file)
# extract_resume_fields(my_resume_file, my_job_title)
# get_job_resources(my_job_title)
if __name__ == '__main__':
    generate_basic_cover_letter(my_company_name, my_job_title, my_resume_file)


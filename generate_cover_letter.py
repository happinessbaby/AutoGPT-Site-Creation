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
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent, Tool, AgentExecutor
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.docstore.wikipedia import Wikipedia
from pathlib import Path
import markdown


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
# Define the function to generate the cover letter
chat = ChatOpenAI(temperature=0.0)
delimiter = "####"

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

    text: '''{text}'''

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=template_string)
    messages = prompt.format_messages(text=resume, 
                                format_instructions=format_instructions)
    
    response = chat(messages)
    personal_info_dict = output_parser.parse(response.content)
    print(personal_info_dict.get('name'))
    return personal_info_dict



def extract_relevant_information(resume_file, job_title):
    loader = TextLoader(resume_file, encoding='utf8')
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])

    job_desciption = get_job_resources(job_title)

    query = f""""Your task is to summarize this text document containing a person's resume.

    List all the fields in the resume  and summarize each field prioritizing information that would help this person get a position as {job_title}.
    
    """
    response = index.query(query)
    print(response)
    return response
    # convert_to_dict(response)



def get_job_resources(job_title):
    docstore = DocstoreExplorer(Wikipedia())
    tools = [
        Tool(
            name = "Search",
            func = docstore.search,
            description= "Search for a term in the docstore."
        ),
        Tool(
            name = "Lookup",
            func = docstore.lookup,
            description = "Lookup a term in the docstore."
        )
    ]
    # tools = load_tools(["wikipedia"], llm=chat)
    agent= initialize_agent(
        tools, 
        chat, 
        agent=AgentType.REACT_DOCSTORE,
        handle_parsing_errors=True,
        verbose = True)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent.agent,
        tools = tools,
        verbose = True,
    )
    question  = f"""Find  out what a {job_title} does and summarize the skills, tasks, responsibilties"""
    agent_executor.run(question)
    # return result

# def convert_to_dict(markdown_table):
#     agent = create_python_agent(
#     chat,
#     tool=PythonREPLTool(),
#     verbose=True    
#     )
#     dict_test = {}
#     agent.run(f"""Convert the markdown table delimited by triple backticks into a python dictionary and save it into the variable {dict_test}. 
              
#               table: ''' {markdown_table}
#               """)

    




def generate_basic_cover_letter(my_company_name, my_job_title, my_resume_file):
    # Read the resume file
    
    try:
        file= my_resume_file.filename
        filename = Path(file).stem
        resume_file = file
    except:
        filename=Path(my_resume_file).stem
        resume_file = my_resume_file
    # resume_filename = my_resume_file.filename
    # with open(resume_filename, 'r') as f:
    #     resume = f.read()
    # print(resume)
    personal_info_dict = extract_personal_information(resume_file)
    relevant_content = extract_relevant_information(resume_file, my_job_title)
    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read
    
    # style_string = f"Dear Hiring Manager, I am writing to express my interest in the {company_name} position at your company. My name is {my_name} ......"
    template_string = """Generate a cover letter for person with name {name} applying to company {company} for the job title {job}. 

      The content you use to generate this personalized cover letter is delimited with {delimiter} characters. content: '''{content}'''"""

    prompt_template = ChatPromptTemplate.from_template(template_string)

    # print(prompt_template.messages[0].prompt.input_variables)

    cover_letter_message = prompt_template.format_messages(
                    # style=cover_letter_template,
                    name = personal_info_dict.get('name'),
                    company = my_company_name,
                    job = my_job_title,
                    content=relevant_content)
    my_cover_letter = chat(cover_letter_message).content
    # cover_letter = get_completion2(template_string)
    
    # Write the cover letter to a file
    with open(f'cover_letter_{filename}.txt', 'w') as f:
        f.write(my_cover_letter)

    return filename




def fine_tune_cover_letter(resume_file):
    # Read the cover letter
    file_name=Path(resume_file).stem
    with open(f'cover_letter_{file_name}.txt', 'r') as f:
        resume = f.read()
    # print(resume)
    
    # fine-tune based on user needs


# Call the function to generate the cover letter

my_job_title = 'MLOps engineer'
my_company_name = 'Facebook'
my_resume_file = 'resume2023v2.txt'
# extract_personal_information(my_resume_file)
# extract_relevant_information(my_resume_file, my_job_title)
get_job_resources(my_job_title)
# if __name__ == '__main__':
#     generate_basic_cover_letter(my_company_name, my_job_title, my_resume_file)


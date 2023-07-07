# Import the necessary modules
import os
import markdown
from openai_api import get_completion, evaluate_response
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.agents import load_tools, initialize_agent, Tool, AgentExecutor
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents import AgentType
from langchain.chains import RetrievalQA
from pathlib import Path
from basic_utils import check_content_safety, read_txt
from langchain_utils import create_wiki_tools, create_document_tools, create_search_tools
from cover_letter_samples import cover_letter_samples_dict
from upgrade_resume import find_similar_jobs
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


def extract_personal_information(resume):

    # with open(resume_file, 'r', errors='ignore') as f:
    #     resume = f.read()

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
    print(personal_info_dict.get('email'))
    return personal_info_dict



## in the future, can add other document tools as resources
def get_job_resources(job_title):

    # wiki_tools = create_wiki_tools()
    # search_tools = create_search_tools()
    # tools = wiki_tools+search_tools
    # SERPAPI has limited searches, for now, skip
    tools = create_wiki_tools()
    
    agent= initialize_agent(
        tools, 
        chat, 
        # agent=AgentType.REACT_DOCSTORE,
        agent="zero-shot-react-description",
        handle_parsing_errors=True,
        verbose = True,
        )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent.agent,
        tools = tools,
        verbose = True,
    )
    query  = f"""Find out what a {job_title} does.
     
       If you cannot find what a {job_title} does, look up responsibilities that are in the same ballpark and find out what they are."""
    # response = agent_executor.run(query)
    # return response
    # TEMPORARY FIX for raise OutputParserException(f"Could not parse LLM Output: {text}") with agent=AgentType.REACT_DOCSTORE
    try:
        response = agent_executor.run(query)
        print(f"Success: {response}")
        return response
    except Exception as e:
        response = str(e)
        # if not response.startswith("Could not parse LLM output: `"):
        #     raise e
        # response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        print(f"Exception RESPONSE: {response}")
        # return response
    


def fetch_cover_letter_samples(job_title):
    table = find_similar_jobs(job_title)

    prompt = f"""Extract the Job Title values in the markdown table: {table}.
    
    Output your answer as a comma separated list. If there is no table, return -1. """

    jobs = get_completion(prompt)
    sample_string = ""
    print(jobs)
    if (jobs!=-1):
        jobs_list = jobs.split(",")
        for job in jobs_list:
            if (cover_letter_samples_dict.get(job)!=None):
                sample = read_txt(cover_letter_samples_dict.get(job))
                sample_string = sample_string + "\n" + f" {delimiter3}\n{sample}\n{delimiter3}" + "\n\nexample:"   
    # print(sample_string)
    return sample_string
        





def generate_basic_cover_letter(my_company_name, my_job_title, read_path=my_resume_file, save_path= "cover_letter.txt"):
    
    resume_content = read_txt(my_resume_file)

    # Get personal information from resume
    personal_info_dict = extract_personal_information(resume_content)

    # Get job description
    job_description = get_job_resources(my_job_title)

    cover_letter_examples = fetch_cover_letter_samples(my_job_title)
    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read

    # Try using a step wise instruction as seen in: https://learn.deeplearning.ai/chatgpt-building-system/lesson/5/chain-of-thought-reasoning

    template_string2 = """Generate a cover letter for a person applying for {job} at {company} using the following information. 

      The content you are to use as reference to create the cover letter is delimited with {delimiter} characters.

        content: {delimiter}{content}{delimiter}. \
    
      Step 1: {delimiter4} Read the content and determine which information in the content is useful and which is not. Usefulness of the information should be based on how close it relates to {job} and the job description, which is delimited with {delimiter2} charaters. \
      
        For example, cooking skills are not related to software development so it is not useful information.

        job description: {delimiter2}{job_description}{delimiter2}. \
    
      Step 2: {delimiter4} Research example cover letters provided. Each example is delimited with {delimiter3} characters.

         Determine which information from Step 1 should be included and which should not based on the quality of information in contributing to a good cover letter.
        
         example: {examples}. \

      Step 3: {delimiter4} Change all personal information to the following. Do not incude them if they are -1 or empty: 

        name: {name}. \

        email: {email}. \

        phone number: {phone}. \
        
        today's date: {date}. \
        
        company they are applying to: {company}. \

        job position they are applying for: {job}. \
    
      Step 4: {delimiter4} Generate the cover letter. Use information you filtered downn in steps 1 through 3. Do not make stuff up. 
    
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
                    job_description = job_description,
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
 
# extract_personal_information(my_resume_file)
# get_job_resources(my_job_title)
# fetch_cover_letter_samples(my_job_title)
if __name__ == '__main__':
    generate_basic_cover_letter(my_company_name, my_job_title)


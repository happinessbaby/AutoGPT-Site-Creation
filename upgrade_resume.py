import os
import markdown
from openai_api import get_completion
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain_utils import get_index, create_google_search_tools, create_custom_llm_agent
from langchain.prompts import ChatPromptTemplate
from basic_utils import read_txt
from langchain.chains.summarize import load_summarize_chain

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

chat = ChatOpenAI(temperature=0.0)
embeddings = OpenAIEmbeddings()
# randomize delimiters from a delimiters list to prevent prompt injection
delimiter = "####"
delimiter1 = "````"
delimiter2 = "////"




def basic_upgrade_resume(resume_file):
    return analyze_resume(resume_file)


def get_resume_tips(query, top=10):
    # look into webpages and summarize content
    tools = create_google_search_tools(top)
    agent_executor = create_custom_llm_agent(chat, tools)
    try:
        response = agent_executor.run(query)
        print(f"Success: {response}")
        return response
    except Exception as e:
        response = str(e)
        print(f"Exception RESPONSE: {response}")
    # Set up the base template
    
    # pages = tool.run(query)
    # print(pages)
    # summaries = []
    # for page in pages:
    #     chain = load_summarize_chain(chat, chain_type="map_reduce")
    #     summary = chain.run(docs)
    #     summaries.append(summary)

    

    

    
    

def analyze_resume(resume_file):

    resume = read_txt(resume_file)
    good_examples = get_resume_tips("good resume examples")
    print(good_examples)
    bad_examples = get_resume_tips("bad resume examples")

    template_string = """" Your task is to determine the strengths and weaknesses of a resume and ways to improve it. 
        
    The resume is delimited with {delimiter} chararacters.
    
    resume: {delimiter}{resume}{delimiter}

    Step 1: Search and extract fields of the resume.  

    Some common resume fields include but not limited to personal information, objective, education, work experience, awards and honors, and skills.

    Step 2: Read through ways to write good resume delimited with {delimiter1} characters. 

    Use the information to determine which fields of the resume in Step 1 are well-written.

    ways to write good resume: {delimiter1}{good_examples}{delimiter1}

    Step 3: Read through bad resume cases delimited with {delimiter2} characters.

    Use the information to determine which fields of the resume in Step 1 are badly written. 

    bad resume bases: {delimiter2}{bad_examples}{delimiter2}

    Step 4: For each badly written fields, give your adviace on how they can be improved, basing your reasoning on Step 2 and Step 3. 


    Use the following format:
        Step 1:{delimiter} <step 1 reasoning>
        Step 2:{delimiter} <step 2 reasoning>
        Step 3:{delimiter} <step 3 reasoning>
        Step 4:{delimiter} <give your advice>

      Make sure to include {delimiter} to separate every step.
    
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    upgrade_resume_message = prompt_template.format_messages(
        resume = resume,
        good_examples = good_examples,
        bad_examples = bad_examples,
        delimiter = delimiter,
        delimiter1 = delimiter1, 
        delimiter2 = delimiter2,
        
    )
    my_advices = chat(upgrade_resume_message).content
    print(my_advices)




# def extract_resume_fields(resume_file):

#     index = get_index(resume_file)

#     query = f"""" Search and extract fields of this resume. 

#     Some common resume fields include but not limited to personal information, objective, education, work experience, awards and honors, and skills.

#     List all the field information in a markdown table.

#     Do not delete any content unless they are absolutely irrelevant. 
    
#     """

#     response = index.query(query)
#     print(response)
#     return response



my_job_title = 'software developer'
my_resume_file = 'resume_samples/resume2023.txt'

if __name__ == '__main__':
    basic_upgrade_resume(my_resume_file)
    # get_resume_tips("summarize good resume examples")
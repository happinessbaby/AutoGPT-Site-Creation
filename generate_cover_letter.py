# Import the necessary modules
import os
from openai_api import get_completion, evaluate_content, check_content_safety
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from basic_utils import read_txt
from common_utils import extract_personal_information, get_web_resources,  get_job_relevancy, retrieve_from_db, get_summary, extract_posting_information, compare_samples, extract_job_title, search_related_samples
from datetime import date
from pathlib import Path
import json
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_utils import create_vectorstore
from multiprocessing import Process, Queue, Value
from base import base


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


#TODO: All the queries can be improved
def generate_basic_cover_letter(my_job_title="", company="", read_path=my_resume_file,  posting_path=""):
    
    userid = Path(read_path).stem
    res_path = os.path.join("./static/cover_letter/", userid + ".txt")

    # Get resume
    resume_content = read_txt(read_path)
    # Get personal information from resume
    personal_info_dict = extract_personal_information(resume_content)
    
    # Get job information from posting link
    job_specification = ""
    if (Path(posting_path).is_file()):
      job_specification = get_summary(posting_path)
      posting = read_txt(posting_path)
      posting_info_dict=extract_posting_information(posting)
      my_job_title = posting_info_dict["job"]
      company = posting_info_dict["company"]

    if (my_job_title==""):
        my_job_title = extract_job_title(resume_content)

    # Get expert advices 
    advice_query = """What are some best practices when writing a cover letter?  """
    advices = retrieve_from_db(advice_query)

    # Get sample comparisons
    query_samples = f""" 
      Research sample cover letters provided. Reference the samples to answer the following questions: 
      what should I put in my cover letter? 
      """
    related_samples = search_related_samples(my_job_title, cover_letter_samples_path)
    if not related_samples:
        related_samples = os.listdir(cover_letter_samples_path)[0]
    practices = compare_samples(related_samples,  query_samples, "cover_letter")
    

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
    

    # Get resume's relevant and irrelevant info for job
    query_relevancy = f"""Determine the relevant and irrelevant information contained in the resume document delimited with {delimiter} characters.

      You are  provided with job specification for an opening position.  Use it as a primarily guidelines when generating your answer. 

      You are also provided with a general job decription of the requirement of {my_job_title}. Use it as a secondary guideline when forming your answer.

      If job specification is not provided, use general job description as your primarily guideline. 

      resume: {delimiter}{resume_content}{delimiter} \n

      job specification: {job_specification}

      general job description: {job_description}

      Generate a list of irrelevant information that should not be included in the cover letter and a list of relevant information that should be included in the cover letter. 

        """
    relevancy = get_job_relevancy(read_path, query_relevancy)
    

    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read

    # Try using a step wise instruction as seen in: https://learn.deeplearning.ai/chatgpt-building-system/lesson/5/chain-of-thought-reasoning

        # The content you are to use as reference to create the cover letter is delimited with {delimiter} characters.

        # Always use this as a context when writing the cover letter. Do not write out of context and do not make anything up. 

        # content: {delimiter}{content}{delimiter}. \n
    template_string2 = """Generate a cover letter for a person applying for {job} at {company} using the following information. 
  

      Step 1: You are given two lists of information delimited with characters. One is irrelevant to applying for {job} and the other is relevant. 

        Use them as a reference when determining what to include and what not to include in the cover letter. 
     
        information list: {relevancy}.  \n

      Step 2: You're given a list of best cover letter practices. Use them when generating the cover letter. 

        best practices: {practices}. \n

      Step 3: You are also given some expert advices. Keep them in mind when generating the cover letter.

        expert advices: {advices}

      Step 3: You're provided with some company informtion. 

        Use it to make the cover letter cater to the company. 

        company information: {company_description}.  \n

    
      Step 4: Change all personal information of the cover letter to the following. Do not incude them if they are -1 or empty: 

        name: {name}. \

        email: {email}. \

        phone number: {phone}. \
        
        today's date: {date}. \
        
        company they are applying to: {company}. \

        job position they are applying for: {job}. \
    
      Step 5: Generate the cover letter using what you've learned in Step 1 through Step 4. Do not make stuff up. 
    
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
                    # content=resume_content,
                    relevancy=relevancy, 
                    practices = practices, 
                    advices = advices,
                    company_description = company_description, 
                    delimiter4 = delimiter4,
    )

    my_cover_letter = llm(cover_letter_message).content
    response = postprocessing(res_path, my_cover_letter)
    create_cover_letter_doc_tool(userid, res_path)
    return response

#TODO
def postprocessing(res_path, response):
    response= response.split(delimiter4)[-1].strip()
    # cut the text to only cover letter
      # transform_chain = TransformChain(
    #     input_variables=["text"], output_variables=["output_text"], transform=transform_func)
    # stream out the answer
    # chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
    # langchain.llm_cache = InMemoryCache()
    expand_qa()
    with open(res_path, 'w') as f:
        try:
            f.write(response)
            print("ALL SUCCESS")
        except Exception as e:
            print("FAILED")
    return response

# process the cover letter and expand on missing information
def expand_qa():
    return None



# tbd: parse the dict json in transform chain before passing into retrievalqaresourcechain
def transform_func(inputs: dict) -> dict:
    
    text = inputs["text"]
    shortened_text = "\n\n".join(text.split("\n\n")[:3])
    return {"output_text": shortened_text}

def preprocessing(json_request):
    
    print(json_request)
    args = json.loads(json_request)
    # if resume doesn't exist, ask for resume
    if ("resume file" not in args or args["resume file"]=="" or args["resume file"]=="<resume file>"):
      return "Can you provide your resume so I can further assist you? "
    else:
      # may need to clean up the path first
        read_path = args["resume file"]
    if ("job" not in args or args["job"] == "" or args["job"]=="<job>"):
        job = ""
    else:
       job = args["job"]
    if ("company" not in args or args["company"] == "" or args["company"]=="<company>"):
        company = ""
    else:
        company = args["company"]
    if ("job post link" not in args or args["job post link"]=="" or args["job post link"]=="<job post link>"):
        posting_path = ""
    else:
        posting_path = args["job post link"]

    res = generate_basic_cover_letter(my_job_title=job, company=company, read_path=read_path, posting_path=posting_path)
    return res
    
def create_cover_letter_doc_tool(userid, res_path):   
    
    name = "faiss_cover_letter"
    description = """This is user's cover letter. 
    Use it as a reference and context when user asks for any questions concerning the preexisting cover letter"""
    create_vectorstore(embeddings, "faiss", res_path, "file",  f"{name}_{userid}")
    # if testing without ui, the below will not run
    chat = base.get_chat()
    chat.add_tools(userid, name, description)

    
def create_cover_letter_generator_tool():
    
    name = "cover_letter_generator"
    parameters = '{{"job":"<job>", "company":"<company>", "resume file":"<resume file>", "job post link": "<job post link>"}}'
    description = f"""Helps to generate a cover letter. Use this tool more than any other tool when user asks to write a cover letter. 
    Input should be JSON in the following format: {parameters} \n
    """
    tools = [
        Tool(
        name = name,
        func = preprocessing,
        description = description, 
        verbose = False,
        )
    ]
    print("Sucessfully created cover letter generator tool. ")
    return tools

  

def test_coverletter_tool():

    tools = create_cover_letter_generator_tool()
    agent= initialize_agent(
        tools, 
        llm=ChatOpenAI(cache=False), 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True,
        )
    response = agent.run(f"""generate a cover letter with following information:
                              job:  \n
                              company:  \n
                              resume file: {my_resume_file} \n
                              job post links: \n                          
                              """)
    return response
  
    
# Call the function to generate the cover letter
 
if __name__ == '__main__':
    generate_basic_cover_letter()
    # test_coverletter_tool()



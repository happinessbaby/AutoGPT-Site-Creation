import os
from openai_api import check_content_safety
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent, create_json_agent
from basic_utils import read_txt
from common_utils import get_web_resources, retrieve_from_db, get_job_relevancy, extract_posting_information, get_summary, extract_fields, get_field_content, extract_job_title, search_related_samples, get_missing_field_items, create_sample_tools
from langchain_utils import retrieve_faiss_vectorstore, create_vectorstore, merge_faiss_vectorstore
# from langchain.text_splitter import MarkdownHeaderTextSplitter
from pathlib import Path
import json
from json import JSONDecodeError
from base import base
from multiprocessing import Process, Queue, Value
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits import JsonToolkit


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# TBD: caching and serialization of llm
llm = ChatOpenAI(temperature=0.0, cache=False)
# llm = OpenAI(temperature=0, top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)
embeddings = OpenAIEmbeddings()
# randomize delimiters from a delimiters list to prevent prompt injection
delimiter = "####"
delimiter1 = "````"
delimiter2 = "////"
delimiter3 = "<<<<"
delimiter4 = "****"

my_job_title = 'accountant'
my_resume_file = 'resume_samples/sample1.txt'
resume_advice_path = './web_data/resume/'
resume_samples_path = './resume_samples/'
posting_path = "./uploads/posting/accountant.txt"


    
def evaluate_resume(my_job_title="", company="", read_path = my_resume_file, posting_path=posting_path):
    
    userid = Path(read_path).stem
    res_path = os.path.join("./static/advice/", Path(read_path).stem + ".txt")
    generated_responses = {}

    resume = read_txt(read_path)

    resume_fields = extract_fields(resume)

    resume_fields = resume_fields.split(",")
    print(f"{resume_fields}")
    generated_responses.update({"resume fields": resume_fields})

    job_specification = ""
    if (Path(posting_path).is_file()):
      job_specification = get_summary(posting_path)
      posting = read_txt(posting_path)
      posting_info_dict=extract_posting_information(posting)
      my_job_title = posting_info_dict["job"]
      company = posting_info_dict["company"]

    if (my_job_title==""):
      my_job_title = extract_job_title(resume)
    generated_responses.update({"job title": my_job_title})
    generated_responses.update({"company name": company})

 
    query_job  = f"""Research what a {my_job_title} does, including details of the common skills, responsibilities, education, experience needed for the job."""
    job_description = get_web_resources(query_job, "google")
    generated_responses.update({"job description": job_description})

    company_description=""
    if (company!=""):
      company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values.                         
                          Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output you don't know"""     
      company_description = get_web_resources(company_query, "wiki")
    generated_responses.update({"company description": company_description})

    related_samples = search_related_samples(my_job_title, resume_samples_path)

    sample_tools = create_sample_tools(related_samples, "resume")

    for field in resume_fields:

      field_dict = { }
      print(f"CURRENT FIELD IS: {field}")
      field_dict[field]= {}
      resume_field_content = get_field_content(resume, field)

      # sample_field_content = ""
      # for sample in related_samples:
      #   sample_field_content += f"sample {field}: " + delimiter1 + get_field_content(sample, field) + delimiter1 + "\n"
      # TODO: update this chain: 0/10
     
      # Get sample comparisons
      query_missing = f""" 

        Compare the {field} in the resume samples with the applicant's field content.

        The applicant's field content is delimited with {delimiter} characters.
      

        applicant's field content: {delimiter}{resume_field_content}{delimiter}

         Answer the following question: 
        
         What is missing in the {field} section of the applicant's resume that are in the samples' {field}?

         Generalize your answer. 

        """
      missing = get_missing_field_items(query_missing, sample_tools)
     
      field_dict[field]["missing field content"]=missing


          
      query_relevancy = f"""Determine the relevant and irrelevant information contained in {field}. 

        You are  provided with job specification for an opening position. 
        
        Use it as a primarily guidelines when generating your answer. 

        You are also provided with a general job decription of the requirement of {my_job_title}. 
        
        Use it as a secondary guideline when forming your answer.

        If job specification is not provided, use general job description as your primarily guideline. 

        field name: {field}

        field content: {resume_field_content}\n

        job specification: {job_specification}\n

        general job description: {job_description} \n

        Generate a list of irrelevant information that should not be included in the resume and a list of relevant information that should be included in {field}. 

          """
      # TODO: Update this chain: 3/10
      relevancy = get_job_relevancy(read_path, query_relevancy)
      field_dict[field]["field relevancy"] = relevancy

      # 6/10, with enough data, this is decent
      # Get expert advices 
      advice_query = f"""What are some best practices when writing {field} for a resume to make it most CTS-friendly?  """
      advices = retrieve_from_db(advice_query)



    q = Queue()
    P = Process(target=postprocessing, args=(generated_responses, res_path, q))
    P.start()
    create_resume_advice_doc_tool(userid, res_path, generated_responses)
    response = q.get()
    P.join()
    return response

    




      # template_string = """" Your task is to analyze and help improve the content of resume field {field}. 

      # The content of the field is delimiter with {delimiter} characters. Always use this as contenxt and do not make things up. 

      # field content: {delimiter}{field_content}{delimiter} \n
      
      # Step 1: You're given some expert advices on how to write {field} . Keep these advices in mind for the next steps.

      #     expert advices: {advices} \n

      # step 2: You are given two lists of information delimited with {delimiter2} characters. One is content to be included in the {field} and the other is content to be removed. 

      #     Use them as to generate your answer. 

      #     information list: {delimiter2}{relevancy}{delimiter2} \n


      # Step 3: You're provided with some company informtion and job specification

      #   Look for ATS-friendly keywords in the job specification and company information to make the resume field cater to the compnay and job position. 

      #   If company information and/or job specifcation do not pertain to the resume field, skip this step. 

      #   company information: {company_description}.  \n

      #   job specification: {job_specification}.    \n

      # Step 4: Based on what you gathered in Step 1 through 3, rewrite the resume field {field}. Do not make up things. 

      # Use the following format:
      #     Step 1:{delimiter4} <step 1 reasoning>
      #     Step 2:{delimiter4} <step 2 reasoning>
      #     Step 3:{delimiter4} <step 3 reasoning>
      #     Step 4:{delimiter4} <rewrite the resume field>


      #   Make sure to include {delimiter4} to separate every step.
      
      # """

      # prompt_template = ChatPromptTemplate.from_template(template_string)
      # upgrade_resume_message = prompt_template.format_messages(
      #     field = field,
      #     field_content = field_content,
      #     job = my_job_title,
      #     advices=advices,
      #     relevancy = relevancy,
      #     company_description = company_description,
      #     job_specification = job_specification, 
      #     delimiter = delimiter,
      #     delimiter1 = delimiter1, 
      #     delimiter2 = delimiter2,
      #     delimiter4 = delimiter4, 
          
      # )
      # my_advice = llm(upgrade_resume_message).content

      # Check potential harmful content in response
      # if (check_content_safety(text_str=my_advice)):   
      #     if (postprocessing(my_advice, res_path)):
      #         create_resume_advice_doc_tool(res_path)

# process response to be outputted to chatbot
def postprocessing(response_dict, res_path, q):
    
    with open(res_path, 'w') as handle:
        try:
            json.dump(response_dict, handle)
            print("ALL SUCCESS")
        except Exception as e:
            print("FAILED")
    with open(res_path, "r") as handle:
       data = json.load(handle)

    json_spec = JsonSpec(dict_=data, max_value_length=4000)
    json_toolkit = JsonToolkit(spec=json_spec)

    json_agent_executor = create_json_agent(
        llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
    )
    
    response = json_agent_executor.run("Convert and condense this JSON document into human readable text")
    print(response)
    q.put(response)
    # TODO: user markdownheadersplitter to split according to delimiters before vs storage

    # cut the text to only cover letter
      # transform_chain = TransformChain(
    #     input_variables=["text"], output_variables=["output_text"], transform=transform_func)
    # stream out the answer
    # chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
    # langchain.llm_cache = InMemoryCache()




# receptionist
def preprocessing(json_request):
    
    print(json_request)
    try:
      args = json.loads(json_request)
    except JSONDecodeError:
      return "Format in JSON and try again." 
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
    res = evaluate_resume(my_job_title=job, company=company, read_path=read_path, posting_path=posting_path)
    return res

def create_resume_evaluator_tool():
    
    name = "resume_evaluator"
    parameters = '{{"job":"<job>", "company":"<company>", "resume file":"<resume file>", "job post link": "<job post link>"}}'
    description = f"""Helps to evaluate a resume. Use this tool more than any other tool when user asks to evaluate, review, help with a resume. 
    Input should be JSON in the following format: {parameters} \n
    (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else) 
    """
    tools = [
        Tool(
        name = name,
        func = preprocessing,
        description = description, 
        verbose = False,
        )
    ]
    print("Succesfully created resume evaluator tool.")
    return tools


def create_resume_advice_doc_tool(userid, res_path, response_dict):   
        
    with open(res_path, 'w') as handle:
      json.dump(response_dict, handle)
    name = "faiss_resume_advice"
    description = """This is user's detailed resume advice. If this tool exists, do not use the 'resume evaluator' tool anymore. 
    Use this tool as a reference to give tailored resume advices. """
    create_vectorstore(embeddings, "faiss", res_path, "file",  f"{name}_{userid}")
    chat = base.get_chat()
    chat.add_tools(userid, name, description)


def test_resume_tool():
    
    tools = create_resume_evaluator_tool()
    agent= initialize_agent(
        tools, 
        llm=ChatOpenAI(cache=False), 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True,
        )
    response = agent.run(f"""evaluate a resume with following information:
                              job:  \n
                              company:  \n
                              resume file: {my_resume_file} \n
                              job post links: \n            
                              """)
    return response
   
   


if __name__ == '__main__':
    # evaluate_resume()
    test_resume_tool()
 
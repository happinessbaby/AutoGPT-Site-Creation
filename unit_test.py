from generate_cover_letter import create_cover_letter_generator_tool
from upgrade_resume import create_resume_evaluator_tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent
from basic_utils import read_txt
from common_utils import (get_web_resources, get_summary, generate_multifunction_response, extract_fields,
                          retrieve_from_db, search_related_samples, create_sample_tools, extract_work_experience, get_field_content, extract_resume_fields)
from langchain.tools import tool
from langchain_utils import retrieve_faiss_vectorstore, create_search_tools
from openai_api import get_completion

resume_file = "resume_samples/sample6.txt"
def generate_random_info(type):
    return None
job = generate_random_info("job")
company = generate_random_info("company")
my_job_title = "software engineer"
delimiter = "####"

# Passed (8/30)
def test_coverletter_tool(resume_file=resume_file, job="data analyst", company="", posting_link="") -> str:

    """ End-to-end testing of the cover letter generator without the UI. """

    tools = create_cover_letter_generator_tool() 
    agent= initialize_agent(
        tools, 
        llm=ChatOpenAI(cache=False), 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True,
        )
    cover_letter = agent.run(f"""generate a cover letter with following information:
                              job: {job} \n
                              company: {company} \n
                              resume file: {resume_file} \n
                              job post links: {posting_link} \n                          
                              """)
    print(f"COVER LETTER: {cover_letter}")


def test_resume_tool(resume_file="./resume_samples/test.txt", job="data analyst", company="", posting_link="") -> str:

    """ End-to-end testing of the resume evaluator without the UI. """
    
    tools = create_resume_evaluator_tool()
    agent= initialize_agent(
        tools, 
        llm=ChatOpenAI(cache=False), 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True,
        )
    resume_advice = agent.run(f"""evaluate a resume with following information:
                              job: {job} \n
                              company: {company} \n
                              resume file: {resume_file} \n
                              job post links: {posting_link}\n            
                              """)
    print(f"RESUME ADVICE: {resume_advice}")


def test_general_job_description(my_job_title="data analyst"):
    job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed. 
   \Keep your answer under 200 words if possible. """
    job_description = get_web_resources(job_query) 
    print(f"JOB DESCRIPTION: {job_description}")


def test_company_description(company="Indeed"):
    company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values.                       
                          Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output -1."""
    company_description = get_web_resources(company_query)
    print(f"COMPANY DESCRIPTION: {company_description}")

# FAIL
def test_work_experience(resume_file="./resume_samples/sample6.txt", my_job_title="software engineer"):
    resume = read_txt(resume_file)
    experience = extract_work_experience(resume, my_job_title)
    print(f"WORK EXPERIENCE LEVEL: {experience}")

# PASS
def test_search_similar_resume(my_job_title = "software engineer"):
    resume_samples_path = "./resume_samples/"
    similar_files=search_related_samples(my_job_title, resume_samples_path)
    print(f"SIMILAR RESUME FILES WRT {my_job_title}: {similar_files}")

# FAIL
def test_extract_fields(resume_file="./resume_samples/resume2023.txt"):
    # fields = extract_fields(read_txt(resume_file))
    extract_resume_fields(read_txt(resume_file))
    # print(f"EXTRACTED RESUME FIELDS ARE: {fields}")


# This part really seems to test LLM's reasoning ability
# PASS
def test_cl_revelancy_query(resume_file=resume_file, my_job_title="data analyst", posting_path=""):

    resume_content = read_txt(resume_file)
    # Search for a job description of the job title
    job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed."""
    job_description = get_web_resources(job_query) 
    if posting_path:
        prompt_template = """Identity the job position, company then provide a summary of the following job posting:
            {text} \n
            Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
        """
        job_specification = get_summary(posting_path, prompt_template)
    else:
        job_specification = ""

     # Get resume's relevant and irrelevant info for job
    query_relevancy = f"""Determine the relevant and irrelevant information contained in the resume document delimited with {delimiter} characters.

      resume: {delimiter}{resume_content}{delimiter} \n

      Generate a list of irrelevant information that should not be included in the cover letter and a list of relevant information that should be included in the cover letter.
       
      Remember to use either job specification or general job description as your guideline. Don't forget to use your tool "search_cover_letter_advice".

      job specification: {job_specification}

      general job description: {job_description} \n

        Your answer should be detailed and only from the resume. Please also provide your reasoning too. 
        
        For example, your answer may look like this:

        Relevant information:

        1. Experience as a Big Data Engineer: using Spark and Python are transferable skills to using Panda in data analysis

        Irrelevant information:

        1. Education in Management of Human Resources is not directly related to skills required for a data analyst 
    
        """  
    tools = create_search_tools("google", 3)
    response = generate_multifunction_response(query_relevancy, tools)
    print(f"RELEVANCY RESPONSE: {response}")


# This part tests the ability to retrieve information from vector stores
# PASS
def test_retrieve_cl(my_job_title="data analyst", education_level="bachelors of science", work_level="entry level"):
    advice_query = f"""Best practices when writing a cover letter for {my_job_title} with the given information:
      highest level of education: {education_level}
      work experience level:  {work_level} """
    # @tool(return_direct=False)
    # def search_cover_letter_advice(query: str) -> str:
    #     """Searches for cover letter writing tips"""
    #     db = retrieve_faiss_vectorstore("faiss_web_data")
    #     retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k":1})
    #     docs = retriever.get_relevant_documents(query)
    #     # reordered_docs = reorder_docs(retriever.get_relevant_documents(subquery_relevancy))
    #     texts = [doc.page_content for doc in docs]
    #     texts_merged = "\n\n".join(texts)
    #     print(f"Successfully used relevancy tool to generate answer: {texts_merged}")
    #     return texts_merged
    
    # generate_multifunction_response(advice_query, [search_cover_letter_advice])
    response = retrieve_from_db(advice_query)
    print(f"RETRIEVAL FROM DB RESPONSE: {response}")


# This seems to test tool using ability and reasoning ability
# BARELY PASSING
def test_cl_samples_query(my_job_title= "accountant"):

    cover_letter_samples_path = "./sample_cover_letters/"
        # Get sample comparisons
    query_samples = f""" 

    Sample cover letters are provided in your tools. Research each one and answer the following question:

      1. What common features these cover letters share?

      """
    related_samples = search_related_samples(my_job_title, cover_letter_samples_path)
    sample_tools = create_sample_tools(related_samples, "cover_letter")
    response = generate_multifunction_response(query_samples, sample_tools)
    print(f"SAMPLES COMPARISON RESPONSE: {response}")

# PASS  
def test_field_retrieval(field="work history"):

    advice_query = f"""ATS-friendly way of writing {field}"""
    advices = retrieve_from_db(advice_query)
    return advices

# FAIL
def test_field_samples_query(resume_file = "./resume_samples/sample1.txt", my_job_title="accountant", field = "work experience"):

    resume_samples_path = "./resume_samples/"
    related_samples = search_related_samples(my_job_title, resume_samples_path)
    print(related_samples)
    sample_tools = create_sample_tools(related_samples, "resume")
    resume_field_content = get_field_content(read_txt(resume_file), field)

    query_missing = f""" 

        Sample resume are provided in your tools. Research each one's field "{field}", if available.

        Answer the following question:

        

        Remember to compare the field specificied and ignore other field content. 

        For example, if you are asked to compare the Education field, ignore other field information from the samples. 

    """
    
    missing_items = generate_multifunction_response(query_missing, sample_tools)
    
    return missing_items

# PASS
def test_field_relevancy_query(resume_file = "./resume_samples/sample5.txt", my_job_title="customer service manager", field = "work history", posting_path=''):

    # relevancy_tools = [search_relevancy_advice]
    tools = create_search_tools("google", 3)
    resume_field_content = get_field_content(read_txt(resume_file), field)

    job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed."""
    job_description = get_web_resources(job_query) 

    if posting_path:
        prompt_template = """Identity the job position, company then provide a summary of the following job posting:
            {text} \n
            Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
        """
        job_specification = get_summary(posting_path, prompt_template)
    else:
        job_specification = ""


    query_relevancy = f"""Determine the relevant and irrelevant information contained in the field content. 

     Generate a list of irrelevant information that should not be included in the cover letter and a list of relevant information that should be included in the resume field.
       
      Remember to use either job specification or general job description as your guideline.

      field name: {field}

      field content: {resume_field_content}\n

      job specification: {job_specification}\n

      general job description: {job_description} \n

      Your answer should be detailed and only from the resume. Please also provide your reasoning too. 
        
      For example, your answer may look like this:

      Relevant information in the Experience field:

      1. Experience as a Big Data Engineer: skills and responsibilities of a Big Data Engineer are transferable to those of a data analyst

      Irrelevant information  in the Experience Field:

      1. Experience as a front desk receptionist is not directly related to the role of a data analyst

      Please do not do anything and output -1 if field name is personal informaion or contains personal information such as name and links. These do not need any evaluation on relevancy

      """
    relevancy = generate_multifunction_response(query_relevancy, tools)
    return relevancy

def test_field_evaluation(resume_file = "./resume_samples/resume2023.txt", field="work history"):

    general_tools = create_search_tools("google", 3)
    resume_field_content = get_field_content(read_txt(resume_file), field)
    advice_query = f"""ATS-friendly way of writing {field}"""
    advices = retrieve_from_db(advice_query)
    query_evaluation = f"""You are given some expert advices on writing {field} section of the resume.
    
    advices: {advices}

    Use these advices to identity the strenghts and weaknesses of the resume content delimited with {delimiter} characters. 
    
    field content: {delimiter}{resume_field_content}{delimiter}. \n
    
    please provide proof of your reasoning. 

    """
    strength_weakness = generate_multifunction_response(query_evaluation, general_tools)
    # strength_weakness = get_completion(query_evaluation)
    return strength_weakness
    

def test_resume_report():
    return None





# test_coverletter_tool()
# test_cl_revelancy_query()
# test_general_job_description()
# test_retrieve_cl()
# test_samples_query()
# test_resume_tool()
# test_work_experience()
# test_field_samples_query()
# test_field_relevancy_query()
# test_field_evaluation()
# test_search_similar_resume()
test_extract_fields()
# test_field_retrieval()
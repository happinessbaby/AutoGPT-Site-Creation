from generate_cover_letter import cover_letter_generator
from upgrade_resume import resume_evaluator
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent
from basic_utils import read_txt
from common_utils import (get_web_resources,extract_education_level,file_loader,
                          retrieve_from_db, search_related_samples, create_sample_tools, extract_work_experience_level,  extract_resume_fields)
from langchain.tools import tool
from langchain_utils import retrieve_faiss_vectorstore, create_search_tools, create_summary_chain, generate_multifunction_response, split_doc, split_doc_file_size, create_refine_chain, create_mapreduce_chain
from openai_api import get_completion
from langchain.chains.summarize import load_summarize_chain
import json
from pathlib import Path

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

    tools = [cover_letter_generator] 
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
    
    tools = [resume_evaluator] + [file_loader]
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
    experience = extract_work_experience_level(resume, my_job_title)
    print(f"WORK EXPERIENCE LEVEL: {experience}")

# Pass
def test_education_level(resume_file="./resume_samples/sample4.txt"):
    resume = read_txt(resume_file)
    education = extract_education_level(resume)
    print(f"EDUCATION: {education}")

# PASS
def test_search_similar_resume(my_job_title = "software engineer"):
    resume_samples_path = "./resume_samples/"
    similar_files=search_related_samples(my_job_title, resume_samples_path)
    print(f"SIMILAR RESUME FILES WRT {my_job_title}: {similar_files}")

# PASS
def test_extract_fields(resume_file="./resume_samples/test.txt"):
    # fields = extract_fields(read_txt(resume_file))
    field_names, field_content = extract_resume_fields(read_txt(resume_file))
    assert isinstance(field_names, list), "field names not a list!"
    assert isinstance(field_content, dict), "field content not a dictionary!"
    print(f"RESUME FIELD NAMES: {field_names}")
    print(f"RESUME FIELD CONTENT: {field_content}")
   


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
        job_specification = create_summary_chain(posting_path, prompt_template)
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
def test_cl_retrieval(my_job_title="data analyst", education_level="bachelors of science", work_level="entry level"):
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
def test_cl_samples_query(my_job_title= "Information Systems Technology Manager"):

    cover_letter_samples_path = "./sample_cover_letters/"
        # Get sample comparisons
    related_samples = search_related_samples(my_job_title, cover_letter_samples_path)
    sample_tools, tool_names = create_sample_tools(related_samples, "cover_letter")
    query_samples = f""" 

    Sample cover letters are provided in your tools. Research {str(tool_names)} and answer the following question: and answer the following question:

      1. What common features these cover letters share?

      """
    response = generate_multifunction_response(query_samples, sample_tools)
    print(f"SAMPLES COMPARISON RESPONSE: {response}")

# PASS  
def test_field_retrieval(field="work history"):

    advice_query = f"""What are some dos and don'ts when writing resume field {field} to make it ATS-friendly?"""
    advices = retrieve_from_db(advice_query)
    return advices

# MISSING CONTENT PART STILL FAILING, REST PASS
def test_field_samples_query(resume_file="./resume_samples/test.txt", my_job_title="accountant",  education_level="Bachelor's of Arts", work_experience="entry level"):

    resume_samples_path = "./resume_samples/"
    related_samples = search_related_samples(my_job_title, resume_samples_path)
    print(related_samples)
    resume = read_txt(resume_file)
    sample_tools, tool_names = create_sample_tools(related_samples, "resume")
    query_sample = f"""  You are an expert resume advisor that specializes in comparing exemplary sample resume. 

    Sample resume are provided in your tools. Research {str(tool_names)} and answer the following question:

       What field information do these resume share, or what content do they have in common? 

    Please do not list your answer but write in complete sentences as an expert resume advisor. 

    Your answer should be general without losing details. For example, you should not compare company names but should compare shared work experiences.

    """  
    commonality = generate_multifunction_response(query_sample, sample_tools)
    advice_query = f"""what to include for resume with someone with {education_level} and {work_experience} for {my_job_title}"""
    # stressing the level of detail in prompt is important
    advices = retrieve_from_db(advice_query)
    query_missing = f""" You are an expert resume advisor that specializes in finding missing fields and content. 
    
            Your job is to find missing items in the resume delimiter with {delimiter} characters using expert suggestions. 

            Ignore all formatting advices and any specific details such as personal details, dates, schools, companies, affiliations, hobbies, etc. 
            
            Your answer should be general enough to allow applicant to fill out the missing content with their own information. 

            Please output only the missing field names and/or missing field content in a list. Do not provide the source. 

            expert suggestion: {advices} \n {commonality} \n

            resume: {delimiter}{resume}{delimiter}. """ 
    
    missing_items = generate_multifunction_response(query_missing, sample_tools)
    print(missing_items)

# PASS
def test_field_relevancy_query( my_job_title="customer service manager", field = "work history", posting_path=''):

    # relevancy_tools = [search_relevancy_advice]
    tools = create_search_tools("google", 3)

    resume_field_content = """Various Positions of Increasing Responsibility and Leadership
            United States Air Force (2018 to present)
             Currently serving as Squadron Operation Superintendent.
             Scheduled to leave the Air Force in September 2021."""

    job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed."""
    job_description = get_web_resources(job_query) 

    if posting_path:
        prompt_template = """Identity the job position, company then provide a summary of the following job posting:
            {text} \n
            Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
        """
        job_specification = create_summary_chain(posting_path, prompt_template)
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

#DECENT, WILL NEED TO UPDATE ADVICE QUERY TO IMPROVE OUTPUT
def test_field_evaluation( field="work history"):

    general_tools = create_search_tools("google", 3)
    resume_field_content = """WALMART, remote contractor, full-time 2022-05 – 2022-10
    BIG DATA ENGINEER
-   Coordinated with other engineers to evaluate and improve low level
    deign of back-end features.

-   Tested methodology with writing and execution of test plans,
    debugging and testing scripts and tools.

-   Updated old code bases to modern development standards, improving
    functionality.

    WALMART, Tuscaloosa, Alabama, full-time 2018-01 – 2018-06

    STORE ASSOCIATE

-   Offered assistance for increased customer satisfaction.

-   Prioritized tasks to meet tight deadlines, pitching in to assist
    others with project duties."""
    advice_query = f"""ATS-friendly way of writing {field}"""
    advices = retrieve_from_db(advice_query)
    query_evaluation = f"""You are given some expert advices on writing {field} section of the resume.
    
    advices: {advices}

    Use these advices to identity the strenghts and weaknesses of the resume content delimited with {delimiter} characters. 
    
    field content: {delimiter}{resume_field_content}{delimiter}. \n
    
    Please provide proof of your reasoning. For example, if there is gap in work history, mark it as a weakness unless there is evidence it should not be considered so.

    """
    strength_weakness = generate_multifunction_response(query_evaluation, general_tools)
    # strength_weakness = get_completion(query_evaluation)
    return strength_weakness
    


def test_refine_chain(directory = "./static/resume_evaluation/test/"):
    
    template = """Write a detailed summary of the following: {text}
        DETAILED SUMMARY: """ 
    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary."
    )
    files = []
    for path in Path(directory).glob('**/*.txt'):
        file = str(path)
        files.append(file)
    print(create_refine_chain(files, template, refine_template))

def test_mapreduce_chain(directory = "./static/resume_evaluation/test/"):

    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes 
    Helpful Answer:"""
    # Reduce
    reduce_template = """The following is set of summaries:
    {doc_summaries}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    files = []
    for path in Path(directory).glob('**/*.txt'):
        file = str(path)
        files.append(file)
    print(create_mapreduce_chain(files, map_template, reduce_template))




# test_coverletter_tool()
# test_resume_tool()

# test_general_job_description()
# test_work_experience()
# test_education_level()

# test_cl_revelancy_query()
# test_cl_retrieval()
test_cl_samples_query()

# test_field_samples_query()
# test_field_relevancy_query()
# test_field_evaluation()
# test_field_retrieval()
# test_search_similar_resume()
# test_extract_fields()
# test_refine_chain()
# test_mapreduce_chain()

from generate_cover_letter import cover_letter_generator, create_cover_letter_generator_tool
from upgrade_resume import resume_evaluator, create_resume_evaluator_tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent
from basic_utils import read_txt
from common_utils import (get_web_resources,extract_education_level,file_loader, binary_file_downloader_html,
                          retrieve_from_db, search_related_samples, create_sample_tools, extract_work_experience_level,  extract_resume_fields)
from langchain.tools import tool
from langchain_utils import retrieve_faiss_vectorstore, create_search_tools, create_summary_chain, generate_multifunction_response, split_doc, split_doc_file_size, create_refine_chain, create_mapreduce_chain
from openai_api import get_completion
from langchain.chains.summarize import load_summarize_chain
from langchain.agents.agent_toolkits import FileManagementToolkit
import json
from pathlib import Path

resume_file = "resume_samples/resume2023v2.txt"
def generate_random_info(type):
    return None
job = generate_random_info("job")
company = generate_random_info("company")
my_job_title = "software engineer"
cover_letter_samples_path = "./sample_cover_letters/"
resume_samples_path = "./resume_samples/"
posting_link = "./uploads/file/data_analyst_SC.txt"
delimiter = "####"

# Passed (8/30)
def test_coverletter_tool(resume_file=resume_file, job="data analyst", company="", posting_link=posting_link) -> str:

    """ End-to-end testing of the cover letter generator without the UI. """

    # tools = [cover_letter_generator] + [binary_file_downloader_html]
    tools = create_cover_letter_generator_tool() + [binary_file_downloader_html]
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



def test_resume_tool(resume_file="./resume_samples/sample1.txt", job="data analyst", company="", posting_link=posting_link) -> str:

    """ End-to-end testing of the resume evaluator without the UI. """
    
    # tools = [resume_evaluator]
    tools = create_resume_evaluator_tool()
    read_tool = FileManagementToolkit(
        root_dir=f"./static/{Path(resume_file).stem}/",
        selected_tools=["read_file"],
    ).get_tools()
    agent= initialize_agent(
        tools + read_tool, 
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


def test_job_specification(my_job_title="data analyst", posting_path=posting_link):
    prompt_template = """Identity the job position, company then provide a summary of the following job posting:
            {text} \n
            Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
        """
    job_specification = create_summary_chain(posting_path, prompt_template)



def test_company_description(company="Southern Company"):
    company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values.                       
                          Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output -1."""
    company_description = get_web_resources(company_query)


# FAIL
def test_work_experience(resume_file="./resume_samples/sample6.txt", my_job_title="software engineer"):
    resume = read_txt(resume_file)
    experience = extract_work_experience_level(resume, my_job_title)

# Pass
def test_education_level(resume_file="./resume_samples/test.txt"):
    resume = read_txt(resume_file)
    education = extract_education_level(resume)


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
# def test_cl_revelancy_query(resume_file=resume_file, my_job_title="data analyst", posting_path=""):

#     resume_content = read_txt(resume_file)
#     # Search for a job description of the job title
#     job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed."""
#     job_description = get_web_resources(job_query) 
#     if posting_path:
#         prompt_template = """Identity the job position, company then provide a summary of the following job posting:
#             {text} \n
#             Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
#         """
#         job_specification = create_summary_chain(posting_path, prompt_template)
#     else:
#         job_specification = ""

#      # Get resume's relevant and irrelevant info for job
#     query_relevancy = f"""Determine the relevant and irrelevant information contained in the resume document delimited with {delimiter} characters.

#       resume: {delimiter}{resume_content}{delimiter} \n

#       Generate a list of irrelevant information that should not be included in the cover letter and a list of relevant information that should be included in the cover letter.
       
#       Remember to use either job specification or general job description as your guideline. Don't forget to use your tool "search_cover_letter_advice".

#       job specification: {job_specification}

#       general job description: {job_description} \n

#         Your answer should be detailed and only from the resume. Please also provide your reasoning too. 
        
#         For example, your answer may look like this:

#         Relevant information:

#         1. Experience as a Big Data Engineer: using Spark and Python are transferable skills to using Panda in data analysis

#         Irrelevant information:

#         1. Education in Management of Human Resources is not directly related to skills required for a data analyst 
    
#         """  
#     tools = create_search_tools("google", 3)
#     response = generate_multifunction_response(query_relevancy, tools)



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



# This seems to test tool using ability and reasoning ability
# BARELY PASSING
# def test_cl_samples_query(my_job_title= "Information Systems Technology Manager"):

#         # Get sample comparisons
#     related_samples = search_related_samples(my_job_title, cover_letter_samples_path)
#     sample_tools, tool_names = create_sample_tools(related_samples, "cover_letter")
#     query_samples = f""" 

#     Sample cover letters are provided in your tools. Research {str(tool_names)} and answer the following question: and answer the following question:

#       1. What common features these cover letters share?

#       """
#     response = generate_multifunction_response(query_samples, sample_tools)




# PASS  
def test_field_retrieval(field="work history"):

    advice_query = f"""What are some dos and don'ts when writing resume field {field} to make it ATS-friendly?"""
    advices = retrieve_from_db(advice_query)



# PASS
def test_field_relevancy( my_job_title="data analyst", field = "skills", posting_path=posting_link):

    # relevancy_tools = [search_relevancy_advice]
    related_samples = search_related_samples(my_job_title, resume_samples_path)
    sample_tools, tool_names = create_sample_tools(related_samples, "resume")

    field_content = """Walmart, remote contractor, full-time					2022-05 – 2022-10
Big Data Engineer 
    • Coordinated across teams to evaluate and improve low level deign of new back-end features that involved upstream streaming and downstream API calls to data analysis sectors
    • Tested methodology with writing and execution of test plans, debugging and testing scripts and tools, which included revamping old code bases to modern development standards
    • Completed all assigned tasks within a timely manner and took on extra responsibilities to meet team goals


Revature									2022-01 - 2022-12
Big Data Engineer
    • Implemented Big Data platforms and tools on the complete ETL process from data ingestion and data query to data storage and data analytic using tools such as Spark, Hadoop, and SQL"""



    job_specification = ""
    job_description = ""
    if posting_path:
        prompt_template = """Identity the job position, company then provide a summary of the following job posting:
            {text} \n
            Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
        """
        job_specification = create_summary_chain(posting_path, prompt_template)
    else:
        job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed."""
        job_description = get_web_resources(job_query) 


    query_relevancy = f"""You are an expert resume advisor. 

     Generate a list of irrelevant information that should not be included in the resume field and a list of relevant information that should be included in the resume field.
       
      Remember to use either job specification or general job description as your guideline. 

      field name: {field}

      field content: {field_content}\n

      job specification: {job_specification}\n

      general job description: {job_description} \n

      Your answer should be detailed and only from the field content. Please also provide your reasoning too as in the following example:
        
            Relevant information in the Experience field:

            1. Experience as a Big Data Engineer: skills and responsibilities of a Big Data Engineer are transferable to those of a data analyst

            Irrelevant information  in the Experience Field:

            1. Experience as a front desk receptionist is not directly related to the role of a data analyst

      """
    relevancy = generate_multifunction_response(query_relevancy, sample_tools)

def test_field_missing(field="Certification", posting_path=posting_link, company="Southern Company"):

    job_specification = ""
    job_description = ""
    if posting_path:
        prompt_template = """Identity the job position, company then provide a summary of the following job posting:
            {text} \n
            Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
        """
        job_specification = create_summary_chain(posting_path, prompt_template)
    else:
        job_query  = f"""Research what a {my_job_title} does and output a detailed description of the common skills, responsibilities, education, experience needed."""
        job_description = get_web_resources(job_query) 

    company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values.                       
                          Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output -1."""
    company_description = get_web_resources(company_query)

    related_samples = search_related_samples(my_job_title, resume_samples_path)
    sample_tools, tool_names = create_sample_tools(related_samples, "resume")
    advice_query = f"""ATS-friendly way of writing {field}"""
    advice = retrieve_from_db(advice_query)

    field_content = """  FOURTHBRAIN 2022-12 – 2023-04

Machine Learning Engineer Program

-   Learned EDA, feature engineering, and classic data modeling
    techniques; built, trained, and evaluated neural networks, including
    CNN, LSTM, and transformers on different data sets and developed
    automated pipeline for deployment on AWS

"""
    
    query_missing_field = f"""  You are an expert resume field advisor. 

     Generate a list of missing information that should be included in the resume field content. 
       
     Remember to use either job specification or general job description and comany description as your guideline. 

      field name: {field}

      field content: {field_content}\n

      job specification: {job_specification}\n

      general job description: {job_description} \n

      company description: {company_description}

      Your answer should be detailed and only from the field content. Please also provide your reasoning too as in the following examples:

            Missing Field Content for Work Experience:

            1. Quantative achievement in project management: no measurable metrics or KPIs to highlight any past achievements. 

     """
            # Missing Field Content for Skills:

            # 1. Technical skills such as SQL and PowerBI to showcase qualification: SQL and PowerBI are listed in the job specification but not in the field content

    relevancy = generate_multifunction_response(query_missing_field, sample_tools)


#DECENT, WILL NEED TO UPDATE ADVICE QUERY TO IMPROVE OUTPUT
def test_field_evaluation( field="work history"):

    general_tools = create_search_tools("google", 3)
    resume_field_content = """Walmart    					                                           2022-05 – 2022-10
Big Data Engineer 
    • Collaborated with cross-functional teams in evaluating and improving low level deign of new back-end features that included streaming services and data analytic
    • Tested methodology with writing and execution of test plans, debugging and testing scripts and tools, and revamped old code bases to modern development standards for improved error handling and data handling to meet the functional programming paradigm 
    • Completed all assigned tasks within a timely manner, aided other team members, and took on extra responsibilities to meet team goals, which helped the team achieve bi-weekly goals more satisfyingly 


Revature									2022-01 - 2022-12
Big Data Engineer
    • Implemented Big Data platforms and tools on the complete ETL process from data ingestion and data query to data storage and data analytic using tools such as Scala, Spark, Hadoop, and SQL
    • Collaborated with other members on Big Data related projects, distributed assignments to team members, and presented results of data analysis to showcase their business values"""
    advice_query = f"""ATS-friendly way of writing {field}"""
    advices = retrieve_from_db(advice_query)
    query_evaluation = f"""You are given some expert advices on writing {field} section of the resume most ATS-friendly.
    
    expert advices: {advices}

    Use these advices to generate a list of weaknesses of the field content and a list of the strengths. 
    
    field content: {resume_field_content}. \n

    Your answer should be detailed and only from the field content. Please also provide your reasoning too as in the following examples:
    
        Weaknesses of Objective:

        1. Weak nouns: you wrote  “LSW” but if the ATS might be checking for “Licensed Social Worker,”

        Strengths of Objective:

        1. Strong adjectives: "Dedicated" and "empathetic" qualities make a good social worker

    Please ignore all formatting advices as formatting should not be part of the assessment.

    """
    strength_weakness = generate_multifunction_response(query_evaluation, general_tools)

    


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

# Pass
def test_cl_combined(posting_path=""):
        # Get sample comparisons
    resume_content = read_txt(resume_file)
    resume_samples_path = "./resume_samples/"
    related_samples = search_related_samples(my_job_title, resume_samples_path)
    sample_tools, tool_names = create_sample_tools(related_samples, "resume")

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
    # Get resume's relevant and irrelevant info for job: few-shot learning works great here
    query_relevancy = f""" You are an expert resume advisor. 
    
     Step 1: Determine the relevant and irrelevant information contained in the resume document delimited with {delimiter} characters.

      resume: {delimiter}{resume_content}{delimiter} \n

      Generate a list of irrelevant information that should not be included in the cover letter and a list of relevant information that should be included in the cover letter.
       
      Remember to use either job specification or general job description as your guideline. 

      job specification: {job_specification}

      general job description: {job_description} \n

        Your answer should be detailed and only from the resume. Please also provide your reasoning too. 
        
        For example, your answer may look like this:

        Relevant information:

        1. Experience as a Big Data Engineer: using Spark and Python are transferable skills to using Panda in data analysis

        Irrelevant information:

        1. Education in Management of Human Resources is not directly related to skills required for a data analyst 


      Step 2:  Sample cover letters are provided in your tools. Research {str(tool_names)} and answer the following question: and answer the following question:

           Make a list of common features these cover letters share. 

        """
    return generate_multifunction_response(query_relevancy, sample_tools)
   
# Pass
def test_resume_missing(my_job_title = "Professor of Geography", highest_education_level="PhD in history", work_experience_level="senior level"):

    related_samples = search_related_samples(my_job_title, resume_samples_path)
    sample_tools, tool_names = create_sample_tools(related_samples, "resume")
    # # Note: retrieval query works best if clear, concise, and detail-dense
    advice_query = f"""what fields to include for resume with someone with {highest_education_level} and {work_experience_level} for {my_job_title}"""
    advice = retrieve_from_db(advice_query)
    resume_fields = ["work experience", "education", "personal information"]

    query_missing = f"""  
  
    Generate a list of missing resume fields suggestions in the job applicant's resume fields given the expert advice. 

    expert advice: {advice}

    job applicant's resume fields: {str(resume_fields)}

    If you believe the applicant's resume fields are enough, output -1. 

    Use your tool if you ever need additional information.

     """ 

    return generate_multifunction_response(query_missing, sample_tools)



# test_coverletter_tool()
# test_resume_tool()

# test_general_job_description()
# test_work_experience()
# test_company_description()
# test_education_level()
# test_job_specification()

# test_cl_revelancy_query()
# test_cl_retrieval()
# test_cl_samples_query()
# test_cl_combined()

# test_field_relevancy()
# test_field_missing()
# test_field_evaluation()
# test_field_retrieval()
# test_search_similar_resume()
# test_extract_fields()
test_resume_missing()
# test_refine_chain()
# test_mapreduce_chain()

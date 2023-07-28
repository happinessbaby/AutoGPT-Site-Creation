import os
from openai_api import evaluate_content, check_content_safety
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from basic_utils import read_txt
from common_utils import search_similar_samples, get_web_resources, retrieve_from_db, get_job_relevancy, extract_fields


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

my_job_title = 'prompt engineer'
my_resume_file = 'resume_samples/sample1.txt'
resume_advice_path = './web_data/resume/'


    
def evaluate_resume(my_job_title, read_path = my_resume_file, res_path="./static/advice/advice.txt"):

    resume = read_txt(read_path)

    resume_fields = extract_fields(resume)
 
    query_advice =  "what makes a bad resume and how to improve"
    resume_advices = retrieve_from_db(resume_advice_path, query_advice)
    
    query_job  = f"""Research what a {my_job_title} does, including details of the common skills, responsibilities, education, experience needed for the job."""
    job_description = get_web_resources(query_job, "google")

    query_relevancy = f"""Determine all the irrelevant information contained in the resume document delimited with {delimiter} characters. 

        You are provided with the skills and responsibilities of {my_job_title}, which is delimited with {delimiter1} charactres, as a reference when forming your answer.

        resume document: {delimiter}{resume}{delimiter}

        skills and responsibilities of {my_job_title}: {delimiter1}{job_description}{delimiter1} \n

        Generate a list of irrelevant information that should not be included in the resume. """

    resume_relevancy = get_job_relevancy(read_path, query_relevancy)

    resume_samples = f"""Research the content of sample resume provided. Each sample is delimited with {delimiter2} characters.

        Compare them with the content delimited with {delimiter2} characters. and look for differences in the content of each field.
        
        sample: {delimiter2}{resume_samples}{delimiter2}"""
    resume_samples = search_similar_samples(embeddings, my_job_title)



    template_string = """" Your task is to analyze the weaknesses of a resume and ways to improve it. 
        
    The resume is delimited with {delimiter1} chararacters.
    
    resume: {delimiter1}{resume}{delimiter1}

    Step 1: Search and extract fields of the resume.  

         Some common resume fields include but not limited to personal information, objective, education, work experience, awards and honors, and skills.

    Step 2: YOu are given a list information delimited with {delimiter4} characters that is considered irrelevant to applying for {job}. 
    
        They should not be included in the resume. 
     
      job_relevancy: {delimiter4}{job_relevancy}{delimiter4}. 

    Step 3: Research sample resume provided. Each sample is delimited with {delimiter2} characters.

        Compare them with resume in Step 1 and list out contents that are in the samples but not in the resume in Step 1.

        These are contents that are probably missing in the resume. 

        sample: {delimiter2}{samples}{delimiter2}

    Step 4: An resume advisor has also given some advices on what makes a bad resume and how to write good resume. The advises are delimited with {delimiter3} characters.
    
        Uses these advices to generate a list of suggestions for fields and content in previous steps. 

        For each badly written field or any missing fields, give your advice on how they can be improved and filled out. 

        advices: {delimiter3}{advices}{delimiter3}


    Use the following format:
        Step 1:{delimiter} <step 1 reasoning>
        Step 2:{delimiter} <step 2 reasoning>
        Step 3:{delimiter} <step 3 reasoning>
        step 4:{delimiter} <step 4 reasoning>


      Make sure to include {delimiter} to separate every step.
    
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    upgrade_resume_message = prompt_template.format_messages(
        resume = resume,
        samples = resume_samples,
        advices = resume_advices,
        job = my_job_title,
        job_relevancy = resume_relevancy,
        delimiter = delimiter,
        delimiter1 = delimiter1, 
        delimiter2 = delimiter2,
        delimiter3 = delimiter3,
        delimiter4 = delimiter4, 
        
    )
    my_advice = llm(upgrade_resume_message).content

    # Check potential harmful content in response
    if (check_content_safety(text_str=my_advice)):   
        # Write the cover letter to a file
        with open(res_path, 'w') as f:
            try:
                f.write(my_advice)
                print("ALL SUCCESS")
            except Exception as e:
                print("FAILED")
                # Error logging



if __name__ == '__main__':
    evaluate_resume(my_job_title)
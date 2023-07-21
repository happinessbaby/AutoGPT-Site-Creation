import os
from openai_api import evaluate_response
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from basic_utils import read_txt, check_content_safety
from common_utils import fetch_samples, get_web_resources
from samples import resume_samples_dict


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

llm = ChatOpenAI(temperature=0.0)
# llm = OpenAI(temperature=0, top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)
embeddings = OpenAIEmbeddings()
# randomize delimiters from a delimiters list to prevent prompt injection
delimiter = "####"
delimiter1 = "````"
delimiter2 = "////"
delimiter3 = "<<<<"
delimiter4 = "****"

my_job_title = 'AI developer'
my_resume_file = 'resume_samples/resume2023v2.txt'


    
def evaluate_resume(my_job_title, read_path = my_resume_file, res_path="advice.txt"):

    resume = read_txt(read_path)
    query  = f"""Find out what a {my_job_title} does and the skills and responsibilities involved. """
    job_description = get_web_resources(llm, query)
    resume_advices = get_web_resources(llm, "what makes a bad resume and how to improve")
    resume_samples = fetch_samples(llm, embeddings, my_job_title, resume_samples_dict)

    template_string = """" Your task is to analyze the weaknesses of a resume and ways to improve it. 
        
    The resume is delimited with {delimiter1} chararacters.
    
    resume: {delimiter1}{resume}{delimiter1}

    Step 1: Search and extract fields of the resume.  

         Some common resume fields include but not limited to personal information, objective, education, work experience, awards and honors, and skills.

    Step 2: Determine if information contained in the fields such as objective, work experience and skills is relevant to {job} and {job}'s job description.

        Generate a list of irrelevant information that should not be included in the resume for {job}. 
     
      job description: {delimiter4}{job_description}{delimiter4}. 

    Step 3: Research sample resume provided. Each sample is delimited with {delimiter2} characters.

        Compare them with resume in Step 1 and list out contents that are in the samples but not in the resume in Step 1.

        These are contents that are probably missing in the resume. 

        sample: {delimiter2}{samples}{delimiter2}

    Step 3: An resume advisor has also given some advices on what makes a bad resume and how to write good resume. The advises are delimited with {delimiter3} characters.
    
        Uses these advices to generate a list of suggestions for fields and content in previous steps. 

        For each badly written field or any missing fields, give your advice on how they can be improved and filled out. 

        advices: {delimiter3}{advices}{delimiter3}


    Use the following format:
        Step 1:{delimiter} <step 1 reasoning>
        Step 2:{delimiter} <step 2 reasoning>
        Step 3:{delimiter} <step 3 reasoning>


      Make sure to include {delimiter} to separate every step.
    
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    upgrade_resume_message = prompt_template.format_messages(
        resume = resume,
        samples = resume_samples,
        advices = resume_advices,
        job = my_job_title,
        job_description = job_description,
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
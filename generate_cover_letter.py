# Import the necessary modules
import os
from openai_api import get_completion, evaluate_response
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from basic_utils import check_content_safety, read_txt
from common_utils import extract_personal_information, fetch_samples, get_web_resources
from samples import cover_letter_samples_dict
from datetime import date

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

# test run defaults, change for yours
my_job_title = 'accountant'
my_resume_file = 'resume_samples/sample1.txt'



def generate_basic_cover_letter(my_job_title, company="abc", read_path=my_resume_file, res_path= "./static/cover_letter.txt"):
    
    resume_content = read_txt(read_path)
    # Get personal information from resume
    personal_info_dict = extract_personal_information(llm, resume_content)
    # Get job description
    query  = f"""Find out what a {my_job_title} does and the skills and responsibilities involved. """
    job_description = get_web_resources(llm, query)
    # Get advices on cover letter
    advices = get_web_resources(llm, "what to include in a good cover letter")
    # Get cover letter examples
    cover_letter_examples = fetch_samples(llm, embeddings, my_job_title, cover_letter_samples_dict)
    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read

    # Try using a step wise instruction as seen in: https://learn.deeplearning.ai/chatgpt-building-system/lesson/5/chain-of-thought-reasoning

    template_string2 = """Generate a cover letter for a person applying for {job} at {company} using the following information. 

      The content you are to use as reference to create the cover letter is delimited with {delimiter} characters.

        content: {delimiter}{content}{delimiter}. \
    
      Step 1: Read the content and determine which information in the content is useful to generate the cover letter. 
      
        Usefulness of the information should be based on how close it is to {job}'s job description. 
      
        The job description is delimited with {delimiter2} characters

        job description: {delimiter2}{job_description}{delimiter2}. \
    
      Step 2: Research example cover letters provided. Each example is delimited with {delimiter3} characters.

        From these examples, determine which information in the content is useful to generate the cover letter.

        Usefulness should be based on how common they appear in these cover letter examples. 
        
         example: {examples}. \
         
      Step 3:  Some expert advices are also provided as basic guidelines. Expert advices are delimited with {delimiter1} characters. 

        Selectively filter down information in Step 1 and Step 2. 

        expert advices: {delimiter1}{advices}{delimiter1}
     

      Step 4: Change all personal information to the following. Do not incude them if they are -1 or empty: 

        name: {name}. \

        email: {email}. \

        phone number: {phone}. \
        
        today's date: {date}. \
        
        company they are applying to: {company}. \

        job position they are applying for: {job}. \
    
      Step 5: Generate the cover letter. 
    
      Use the following format:
        Step 1:{delimiter4} <step 1 reasoning>
        Step 2:{delimiter4} <step 2 reasoning>
        Step 3:{delimiter4} <step 3 reasoning>
        Step 4:{delimiter4} <step 4 reasoning>
        Step 5:{delimiter4} <the cover letter you generate>

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
                    company = company,
                    job = my_job_title,
                    content=resume_content,
                    advices = advices,
                    job_description = job_description,
                    examples = cover_letter_examples,
                    delimiter = delimiter,
                    delimiter1 = delimiter1, 
                    delimiter2 = delimiter2,
                    delimiter3 = delimiter3,
                    delimiter4 = delimiter4
    )
    # FOR THE FUTURE, THIS LLM HERE CAN BE CUSTOMLY TRAINED
    my_cover_letter = llm(cover_letter_message).content

    my_cover_letter= my_cover_letter.split(delimiter4)[-1].strip()

    # Check potential harmful content in response
    if (check_content_safety(text_str=my_cover_letter)):   
        # Validate cover letter
        if (evaluate_response(my_cover_letter)):
            # Write the cover letter to a file
            with open(res_path, 'w') as f:
                try:
                    f.write(my_cover_letter)
                    print("ALL SUCCESS")
                except Exception as e:
                    print("FAILED")
                    # Error logging
    
        

# Call the function to generate the cover letter
 
if __name__ == '__main__':
    generate_basic_cover_letter(my_job_title)



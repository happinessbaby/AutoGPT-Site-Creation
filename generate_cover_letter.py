# Import the necessary modules
import os
from openai_api import get_completion1, get_completion2
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Define the function to generate the cover letter
chat = ChatOpenAI(temperature=0.0)

def generate_cover_letter(my_name, resume_file, my_company):
    # Read the resume file
    with open(resume_file, 'r') as f:
        resume = f.read()
    # print(resume)
    
    # Extract the relevant information from the resume
    # name = resume.split('\n')[0]
    # email = resume.split('\n')[1]
    # phone = resume.split('\n')[2]
    
    # Scrape out personal information
    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read
    
    # style_string = f"Dear Hiring Manager, I am writing to express my interest in the {company_name} position at your company. My name is {my_name} ......"

    template_string = """Generate a cover letter for person with name {name} applying to company {company}. 
      The content you use to generate this personalized cover letter is delimited by triple backticks. content: '''{content}'''"""

    prompt_template = ChatPromptTemplate.from_template(template_string)

    print(prompt_template.messages[0].prompt.input_variables)

    cover_letter_message = prompt_template.format_messages(
                    # style=cover_letter_template,
                    name = my_name,
                    company = my_company,
                    content=resume)
    cover_letter = chat(cover_letter_message).content
    # cover_letter = get_completion2(template_string)
    
    
    # Write the cover letter to a file
    with open(f'cover_letter_{resume_file}.txt', 'w') as f:
        f.write(cover_letter)

# Call the function to generate the cover letter
my_name = 'Yueqi Peng'
resume_file = 'resume2023v2.txt'
company_name = 'XYZ Corporation'
generate_cover_letter(my_name, resume_file, company_name)


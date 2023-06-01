# Import the necessary modules
import os
from openai_api import get_completion1, get_completion2

# Define the function to generate the cover letter

def generate_cover_letter(my_name, resume_file, company_name):
    # Read the resume file
    with open(resume_file, 'r') as f:
        resume = f.read()
    print(resume)
    
    # Extract the relevant information from the resume
    # name = resume.split('\n')[0]
    # email = resume.split('\n')[1]
    # phone = resume.split('\n')[2]
    
    # Scrape out personal information
    
    # Use an LLM to generate a cover letter that is specific to the resume file that is being read
    
    template = f"Dear Hiring Manager, I am writing to express my interest in the {company_name} position at your company. My name is {my_name} ......"

    prompt = f"""You're to generate a cover letter for person with name {my_name} applying to company {company_name}. This is the template: {template}. This is the content based on which you're to generate a personalized cover letter: '''{resume}'''"""

    # print(prompt)

    cover_letter = get_completion2(prompt)
    
    
    # Write the cover letter to a file
    with open(f'cover_letter_{resume_file}.txt', 'w') as f:
        f.write(cover_letter)

# Call the function to generate the cover letter
my_name = 'Yueqi Peng'
resume_file = 'resume2023v2.txt'
company_name = 'XYZ Corporation'
generate_cover_letter(my_name, resume_file, company_name)


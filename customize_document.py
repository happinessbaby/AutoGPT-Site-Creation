from basic_utils import read_txt
from openai_api import get_completion
from openai_api import get_completion, get_completion_from_messages
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate,  StringPromptTemplate
from langchain.agents import load_tools, initialize_agent, Tool, AgentExecutor
from langchain.document_loaders import CSVLoader, TextLoader
from pathlib import Path
from basic_utils import read_txt, convert_to_txt
from langchain_utils import create_search_tools, generate_multifunction_response
from langchain import PromptTemplate
from common_utils import get_generated_responses, get_web_resources
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from typing import Any, List, Union, Dict
from langchain.docstore.document import Document
from langchain.tools import tool
import json
from json import JSONDecodeError
import faiss
import asyncio
import random
import base64
from datetime import date
from feast import FeatureStore
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


def customize_document(file="", about_me="", file_type=""):

    file = read_txt(file)
    if file_type == "personal statement":
        generated_info_dict=get_generated_responses(personal_statement=personal_statement, about_me=about_me)

        institution_description=generated_info_dict["institution description"] 
        program_description = generated_info_dict["program description"]

        query = f""" You are to help Human polish a personal statement. 
        Prospective employers and universities may ask for a personal statement that details qualifications for a position or degree program.
        You are provided with a personal statement and various pieces of information that you may use.
        personal statement: {personal_statement}
        institution description: {institution_description}
        program description: {program_description}

        please consider blending these information into the existing personal statement. Use the same tone and style when inserting information.
        

        """
        tools = create_search_tools("google", 3)
        response = generate_multifunction_response(query, tools)
    return response




@tool
def document_customize_writer(json_request:str) -> str:

    """ Helps customize, personalize, and improve personal statement, cover letter, or resume. 
    Input should be a single string strictly in the following JSON format: '{{"file":"<file>", "file type":"<file type>", "about me":"<about me>"}}' 
    file type should be either "personal statement", "cover letter", or "resume \n
    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else)"""
    
    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    
    if ("file" not in args or args["file"]=="" or args["file"]=="<file>"):
        return """stop using or calling the document_improvement_writer tool. Ask user to upload their file instead. """
    else:
        file = args["file"]
    if ("about me" not in args or args["about me"] == "" or args["about me"]=="<about me>"):
        about_me = ""
    else:
        about_me = args["about me"]
    if ("file type" not in args or args["file type"] == "" or args["file type"]=="<file type>"):
        file_type = ""
    else:
        file_type = args["file type"]
    print(file, about_me, file_type)
    customize_document(file=file, about_me=about_me, file_type=file_type)



def create_document_customize_writer_tool() -> List[Tool]:

    """ Agent tool that calls the function that writes personal statement """

    name = "personal_statement_writer"
    parameters = '{{"file":"<file>", "file type":"<file type>", "about me":"<about me>"}}'
    description = f""" Helps customize, personalize, and improve personal statement, cover letter, or resume. 
    Input should be a single string strictly in the following JSON format: {parameters}
    file type should be either "personal statement", "cover letter", or "resume \n
    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else)"""
    tools = [
        Tool(
        name = name,
        func = process_document,
        description = description, 
        verbose = False,
        handle_tool_error=True,
        )
    ]
    print("Succesfully created interview initiator tool.")
    return tools

def process_document(json_request: str) -> str:

    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    
    if ("file" not in args or args["file"]=="" or args["file"]=="<file>"):
        return """stop using or calling the document_improvement_writer tool. Ask user to upload their file instead. """
    else:
        file = args["file"]
    if ("about me" not in args or args["about me"] == "" or args["about me"]=="<about me>"):
        about_me = ""
    else:
        about_me = args["about me"]
    if ("file type" not in args or args["file type"] == "" or args["file type"]=="<file type>"):
        file_type = ""
    else:
        file_type = args["file type"]
    print(file, about_me, file_type)
    customize_document(file=file, about_me=about_me, file_type=file_type)


if __name__=="__main__":
    personal_statement = "./uploads/file/personal_statement.txt"
    customize_document(about_me="i want to apply for a MSBA program at university of louisville", file = personal_statement, file_type="personal statement")
        
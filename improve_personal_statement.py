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


def improve_personal_statement(personal_statement="", about_me=""):

    personal_statement = read_txt(personal_statement)
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





def create_personal_statement_writer_tool() -> List[Tool]:

    """ Agent tool that calls the function that writes personal statement """

    name = "personal_statement_writer"
    parameters1 = '{{"personal statement file":"<personal statement file>", "about me":"<about me>"}}'
    description = f"""
        Use this tool whenever Human wants your help to write a personal statement or help them check their current one. Do not use this tool for study purposes or answering interview questions. 
        Input should be a single string strictly in the following JSON format: {parameters1} \n 
        (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else) \n"""
    tools = [
        Tool(
        name = name,
        func = process_personal_statement,
        description = description, 
        verbose = False,
        handle_tool_error=True,
        )
    ]
    print("Succesfully created interview initiator tool.")
    return tools

def process_personal_statement(json_request: str) -> str:

    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    
    if ("personal statement file" not in args or args["personal statement file"]=="" or args["personal statement file"]=="<personal statement file>"):
        return """stop using or calling the personal_statement_writer tool. Ask user to upload their personal statement. """
    else:
        personal_statement = args["personal statement file"]
        about_me = args["about me"]
        improve_personal_statement(personal_statement=personal_statement, about_me=about_me)

if __name__=="__main__":
    personal_statement = "./uploads/file/personal_statement.txt"
    improve_personal_statement(about_me="i want to apply for a MSBA program at university of louisville", personal_statement=personal_statement)
        
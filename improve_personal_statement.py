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
from common_utils import get_generated_responses, extract_personal_statement_information, get_web_resources
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





def create_personal_statement_writer_tool() -> List[Tool]:

    """ Agent tool that calls the function that writes personal statement """

    name = "personal_statement_writer"
    parameters1 = '{{"personal statement":"<personal statement>", "institution":"<institution>", "program":"<program>"}}'
    description = f"""
        Use this tool whenever Human wants your help to write a personal statement or help them check their current one. Do not use this tool for study purposes or answering interview questions. 
        Input should be a single string strictly in the following JSON format: {parameters1} \n 
        (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else) \n"""
    tools = [
        Tool(
        name = name,
        func = improve_personal_statement,
        description = description, 
        verbose = False,
        handle_tool_error=True,
        )
    ]
    print("Succesfully created interview initiator tool.")
    return tools

def improve_personal_statement(json_request: str) -> str:

    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    
    if ("personal statement" not in args or args["personal statement"]=="" or args["personal statement"]=="<personal statement>"):
        return """stop using or calling the personal_statement_writer tool. Ask user to upload their personal statement. """
    else:
        personal_statement = args["personal statement"]
        personal_statement = read_txt(personal_statement)
        personal_statement_dict = extract_personal_statement_information(personal_statement)
        institution = personal_statement_dict["institution"] or ""
        program = personal_statement_dict["program"] or ""
        if institution != "":
            institution_query = f""" Research {institution}'s culture, mission, and values.                       
                          Look up the exact name of the institution. If it doesn't exist or the search result does not return an institution output -1."""
            institution_description = get_web_resources(institution_query)
        if program!="":
            pursuit_query = f"""Research the degree program in the institution provided below. 
            Find out what this program in the institution involves, and what's special about the program, and why it's worth pursuing.     
                program: {program} \n
                institution: {institution} \n  
            If institution is not available, research the general program itself.
            """
            pursuit_description = get_web_resources(pursuit_query)    
            
        query = f""" You are to help Human polish a personal statement. 
        Prospective employers and universities may ask for a personal statement that details qualifications for a position or degree program.
        You are provided with a personal statement and various pieces of information that you may add to the personal statement. 
        personal statement: {personal_statement}
        institution description: {institution_description}
        pursuit description: {pursuit_description}

        If you think whatever you're provided with is not enough, ask Human for more information.

        DO NOT MAKE ANYTHING UP. If you lack information to write a good personal statement, just ask Human to provde it for you. 

        """
        tools = create_search_tools("google", 3)
        response = generate_multifunction_response(query, tools)
        return response


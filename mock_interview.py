from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.cache import InMemoryCache
import langchain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json
from json import JSONDecodeError

import os
from pathlib import Path
from typing import Any
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from common_utils import file_loader, check_content
from langchain_utils import (create_vectorstore, create_summary_chain,
                             retrieve_redis_vectorstore, split_doc, CustomOutputParser, CustomPromptTemplate, create_vs_retriever_tools,
                             create_retriever_tools, retrieve_faiss_vectorstore, merge_faiss_vectorstore, handle_tool_error, create_search_tools, create_wiki_tools)
# from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool, load_tools
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict, AgentAction
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.docstore import InMemoryDocstore
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.tools.human.tool import HumanInputRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import get_openai_callback, StdOutCallbackHandler, FileCallbackHandler
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.vectorstores import FAISS
# from feast import FeatureStore
import pickle
import json
import langchain
import faiss
from loguru import logger
from langchain.evaluation import load_evaluator
from basic_utils import convert_to_txt, read_txt
from openai_api import get_completion
from langchain.schema import OutputParserException
from multiprocessing import Process, Queue, Value
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from typing import List, Dict
from json import JSONDecodeError
from langchain.tools import tool, format_tool_to_openai_function
import re
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.tools.file_management.read import ReadFileTool
from langchain.cache import InMemoryCache
from langchain.tools import StructuredTool


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

memory_key="chat_history"
memory_max_token = 500
log_path = os.environ["LOG_PATH"]
# set recording parameters
duration = 5 # duration of each recording in seconds
fs = 44100 # sample rate
channels = 1 # number of channel
# Code for audio part: https://github.com/VRSEN/langchain-agents-tutorial/blob/main/main.py


class InterviewController():

    llm = ChatOpenAI(streaming=True,  callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
    # langchain.llm_cache = InMemoryCache()
    embeddings = OpenAIEmbeddings()
    # chat_memory = ReadOnlySharedMemory(memory=chat_memory)
    # initialize new memory (shared betweeen interviewer and grader_agent)
    interview_memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm, input_key="input", max_token_limit=memory_max_token)


    def __init__(self, userid, additional_prompt_info):
        self.userid = userid
        self.mode = "interview"
        self.additional_interview_info = additional_prompt_info
        self._initialize_log()
        self._initialize_interview_agent()
        self._initialize_interview_grader()

    def _initialize_log(self) -> None:
         
        """ Initializes log: https://python.langchain.com/docs/modules/callbacks/filecallbackhandler """

         # initialize file callback logging
        logfile = log_path + f"{self.userid}.log"
        self.handler = FileCallbackHandler(logfile)
        logger.add(logfile,  enqueue=True)
        # Upon start, all the .log files will be deleted and changed to .txt files
        for path in  Path(log_path).glob('**/*.log'):
            file = str(path)
            file_name = path.stem
            if file_name != self.userid: 
                # convert all non-empty log from previous sessions to txt and delete the log
                if os.stat(file).st_size != 0:  
                    convert_to_txt(file, f"./log/{file_name}.txt")
                os.remove(file)


    def _initialize_interview_agent(self) -> None:


        """ Initialize interviewer agent, a Conversational Retrieval Agent: https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents

        Args: 

            json_request (str): input argument from human's question, in this case the interview topic that may be contained in the Human input.

        """
 
           
        # interview_stopper_tool = self.ends_interview_tool()h.

        wiki_tool = create_wiki_tools()
        self.interview_tools = wiki_tool
        # load vector store
        vs_directory = f"./faiss/{self.userid}/"
        try:
            subfolders= [f.path for f in os.scandir(vs_directory) if f.is_dir()]
            for dirname in list(subfolders):
                vs = FAISS.load_local(dirname, self.embeddings)
                # suffix = dirname.rfind("_")
                # tool_name = "search_" + dirname[:suffix].removeprefix(vs_directory)
                tool_name = "search_" + dirname.removeprefix(vs_directory).removesuffix(f"_{self.userid}")
                tool_description =  """Useful for generating personalized interview questions and answers. 
                    Use this tool more than any other tool during a mock interview session when asking personal questions such as work experience, personal projects, tell me about yourself.
                    Do not use this tool to load any files or documents.  """ 
                tools = create_vs_retriever_tools(vs, tool_name, tool_description)
                self.interview_tools+=tools
        except FileNotFoundError:
            pass
        
        # try:
        #     self.interview_tools
        # except AttributeError:
        #     self.interview_tools = wiki_tool
        # else:
        #     self.interview_tools += wiki_tool     


            # Human may also have asked for a specific interview topic: {topic}
        #initialize interviewer agent

        template =   f"""
            You are an AI job interviewer. The following, if available, are things pertaining to the interview. Generate your interview questions from them. 
            
           {self.additional_interview_info}

            The interview content is contained in the tool "search_interview_material", if available. Generate your interview questions from this tool.

            Personal information about the applicant is contained in the tool "search_resume", if available.

            As an interviewer, you do not need to assess Human's response to your questions. Their response will be sent to a professional grader.         

            Sometimes you will be provided with the professional grader's feedback. They will be handed out to the Human at the end of the session. You should ignore them. 

            Always remember your role as an interviewer. Unless you're told to stop interviewing, you should not stop asking interview questions.

            If the Human is asking about other things instead of answering an interview question, please steer them back to the interview process.

            Please ask your interview question now!

           """
        
        system_message = SystemMessage(
        content=(
          template
        )
        )
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
            )

        print(f"INTERVIEW AGENT TOOLS: {[tools.name for tools in self.interview_tools]}")
        agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.interview_tools, prompt=prompt)

        # messages = chat_prompt.format_messages(
        #           grader_feedback = self.grader_feedback,
        #         instructions = self.instructions
        # )

        # prompt = OpenAIFunctionsAgent.create_prompt(
        #         # system_message=system_msg,
        #         extra_prompt_messages=messages,
        #     )
        # agent = OpenAIFunctionsAgent(llm=llm, tools=study_tools, prompt=prompt)

        self.interview_agent = AgentExecutor(agent=agent,
                                    tools=self.interview_tools, 
                                    memory=self.interview_memory, 
                                    # verbose=True,
                                    return_intermediate_steps=True,
                                    handle_parsing_errors=True,
                                    callbacks = [self.handler])
        




    def _initialize_interview_grader(self) -> None:


        """ Initialize interview grader agent, a Conversational Retrieval Agent: https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents """

        system_message = SystemMessage(
        content=(

          f"""You are a professional interview grader who grades the quality of responses to interview questions. 
          
          Access your memory and retrieve the very last piece of the conversation, if available.

          Determine if the AI has asked an interview question. If it has, you are to grade the Human input based on how well it answers the question.

          Otherwise, respond with the phrase "skip" only.

          The following, if available, are things pertaining to the interview.
            
           {self.additional_interview_info}

           The interview content is contained in the tool "search_interview_material", if available.

            Personal information about the applicant is contained in the tool "search_resume", if available.
        
            Use these tools to search for the correct answer.

          If the answer cannot be found in the your search, use other tools or answer to your best knowledge. 

          Remember, the Human may not know the answer or may have answered the question incorrectly. Therefore it is important that you provide an informative feedback to the Human's response in the format:

          Positive Feedback: <in which ways the Human answered the question well>

          Negative Feedback: <in which ways the Human failed to answer the question>
        
            """
        #   Your feedback should take both the correct answer and the Human's response into consideration. When the Human's response implies that they don't know the answer, provide the correct answer in your feedback.
        )
        )
        prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
        )
        agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.interview_tools, prompt=prompt)
        self.grader_agent = AgentExecutor(agent=agent, 
                                        tools=self.interview_tools, 
                                        memory=self.interview_memory, 
                                        # verbose=True,
                                        return_intermediate_steps=True, 
                                        handle_parsing_errors=True,
                                        callbacks = [self.handler])
        

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff between retries
        stop=stop_after_attempt(5)  # Maximum number of retry attempts
    )
    def askAI(self, user_input: str, callbacks=None,) -> str:

        """ Main function that processes all agents' conversation with user.
         
        Args:

            userid (str): session id of user

            user_input (str): user question or response

        Keyword Args:

            callbacks: default is None

        Returns:

            Answer or response by AI (str)  
            
         """

        try:    
            # BELOW IS USED WITH CONVERSATIONAL RETRIEVAL AGENT (grader_agent and interviewer)
            # if (update_instruction):
            #     instruction = self.askMetaAgent()
            #     print(instruction) 
            grader_feedback = self.grader_agent({"input":user_input}).get("output", "")
            # print(f"GRADER FEEDBACK: {grader_feedback}")
            print(f"User Voice Input: {user_input}")
            response = self.interview_agent({"input":user_input})
            response = response.get("output", "sorry, something happened, try again.")        
            # response = self.interview_agent({"input":user_input})    
            # if (evaluate_result):
            #     evalagent_q = Queue()
            #     evalagent_p = Process(target = self.askEvalAgent, args=(response, evalagent_q, ))
            #     evalagent_p.start()
            #     evalagent_p.join()
            #     evaluation_result = evalagent_q.get()
            #     # add evaluation and instruction to log
            #     self.update_meta_data(evaluation_result)
            
            # convert dict to string for chat output
        # let instruct agent handle all exceptions with feedback loop
        except Exception as e:
            print(f"ERROR HAS OCCURED IN ASKAI: {e}")
            error_msg = str(e)
            # needs to get action and action input before error and add it to error message
            # if (update_instruction):
            #     query = f""""Debug the error message and provide Instruction for the AI assistant: {error_msg}
            #         """        
            #     instruction = self.askMetaAgent(query)
                # self.update_instructions(feedback)
            # if evaluate_result:
            #     self.update_meta_data(error_msg)
            self.askAI(user_input, callbacks)        

        # pickle memory (sanity check)
        # with open('conv_memory/' + userid + '.pickle', 'wb') as handle:
        #     pickle.dump(self.chat_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print(f"Sucessfully pickled conversation: {chat_history}")
        return response
    
    
    
    # def update_tools(self, tools: List[Tool]) -> None:

    #     """ Updates interview tools, which are study material or resume converted to vectore store retriever as tools """
    #     try:
    #         self.interview_tools
    #     except AttributeError:
    #         self.interview_tools = tools
    #     else:
    #         self.interview_tools += tools
    #     print(f"Succesfully updated tool {[t.name for t in tools]}for interview agent.")
    
        

    # def ends_interview_tool(self) -> List[Tool]:

    #     """ Agent tool used to end the interview session."""
        
    #     name = "interview_stopper"
    #     parameters = '{{"end interivew": "<end interview>"}}'
    #     description = f"""Ends the interview process.Use this whenever user asks to end the interview session or go back to the main chat. 
    #         Do not use it in any other circumstances.  
    #         Input should be empty  \n
    #          """
    #     tools = [
    #         Tool(
    #         name = name,
    #         func = self.interview_stopper,
    #         description = description, 
    #         verbose = False,
    #         handle_tool_error=handle_tool_error,
    #         )
    #     ]
    #     print("Succesfully created interview initiator tool.")
    #     return tools


    # def interview_stopper(self, json_request:str) -> str:

    #     # try:
    #     #     args = json.loads(json_request)
    #     #     request = args["user request"]
    #     # except Exception:
    #     #     request = "please inform user that interivew session has ended"
    #     self.mode = "chat"

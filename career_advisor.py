
import os
from pathlib import Path
from typing import Any
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, LLMChain
# from langchain.agents import ConversationalChatAgent, Tool, AgentExecutor
# from basic_utils import read_txt
from langchain_utils import (create_QASource_chain, create_qa_tools, create_vectorstore,
                             retrieve_redis_vectorstore, split_doc, CustomOutputParser, CustomPromptTemplate,
                             create_db_tools, retrieve_faiss_vectorstore, merge_faiss_vectorstore, handle_tool_error, create_search_tools, create_wiki_tools)
# from langchain.prompts import BaseChatPromptTemplate
# from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain.schema import AgentAction, AgentFinish, HumanMessage
# from typing import List, Union
# import re
# from langchain import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool
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

# from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
# from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
# from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
# from langchain.schema.messages import SystemMessage
# from langchain.prompts import MessagesPlaceholder
# from langchain.agents import AgentExecutor
# from langchain.agents.agent_toolkits import (
#     create_vectorstore_agent,
#     VectorStoreToolkit,
#     create_vectorstore_router_agent,
#     VectorStoreRouterToolkit,
#     VectorStoreInfo,
# )
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
from generate_cover_letter import  create_cover_letter_generator_tool
from upgrade_resume import create_resume_evaluator_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from typing import List
from json import JSONDecodeError
from common_utils import create_file_loader_tool, create_debug_tool, create_search_all_chat_history_tool
from langchain.tools import tool, format_tool_to_openai_function
import re
from tenacity import retry, wait_exponential, stop_after_attempt



from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
log_path = os.environ["LOG_PATH"]
# debugging langchain: very useful
langchain.debug=True 
# The result evaluation process slows down chat by a lot, unless necessary, set to false
evaluate_result = False
# The instruction update process is still being tested for effectiveness
update_instruction = True
delimiter = "####"
word_count = 100
memory_max_token = 500
memory_key="chat_history"



### ERROR HANDLINGS: 
# for tools, will use custom function for all errors. 
# for chains, top will be langchain's default error handling set to True. second layer will be debub_tool with custom function for errors that cannot be automatically resolved.


        


class ChatController():

    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-0613", streaming=True,)
    embeddings = OpenAIEmbeddings()
    chat_memory = ConversationBufferMemory(llm=llm, memory_key=memory_key, return_messages=True, input_key="input", output_key="output", max_token_limit=memory_max_token)
    # chat_memory = ReadOnlySharedMemory(memory=chat_memory)
    # initialize new memory (shared betweeen study_agent and grader_agent)
    study_memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm, input_key="input", max_token_limit=memory_max_token)
    # retry_decorator = _create_retry_decorator(llm)


    def __init__(self, userid):
        self.userid = userid
        self._initialize_log()
        self._initialize_chat_agent()
        if update_instruction:
            self._initialize_meta_agent()
        


    def _initialize_chat_agent(self):
    
        # initialize tools
        # Specific purpose tools
        cover_letter_tool = create_cover_letter_generator_tool()
        resume_advice_tool = create_resume_evaluator_tool()
        study_companion_tool = self.initiate_study_companion_tool()
        # General purpose tools
        file_loader_tool = create_file_loader_tool()
        redis_store = retrieve_redis_vectorstore(self.embeddings, "index_web_advice")
        redis_retriever = redis_store.as_retriever()
        general_tool_description = """This is a general purpose database. Use it to answer general job related questions. 
        Prioritize other tools over this tool. """
        general_tool= create_db_tools(self.llm, redis_retriever, "redis_general", general_tool_description)
        # web reserach: https://python.langchain.com/docs/modules/data_connection/retrievers/web_research
        search = GoogleSearchAPIWrapper()
        embedding_size = 1536  
        index = faiss.IndexFlatL2(embedding_size)  
        vectorstore = FAISS(self.embeddings.embed_query, index, InMemoryDocstore({}), {})
        web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=vectorstore,
            llm=self.llm, 
            search=search, 
        )
        web_tool_description="""This is a web research tool. Use it only when you cannot find answers elsewhere. Always return source information."""
        web_tool = create_db_tools(self.llm, web_research_retriever, "faiss_web", web_tool_description)
        # basic search tool 
        search_tool = create_search_tools("google", 5)
        # wiki tool
        wiki_tool = create_wiki_tools()
        # debug tool
        debug_tool = create_debug_tool()
        # gather all the tools together
        self.tools = general_tool + cover_letter_tool + resume_advice_tool + web_tool + study_companion_tool + search_tool + file_loader_tool + wiki_tool + debug_tool


        # initialize evaluator
        if (evaluate_result):
            self.evaluator = load_evaluator("trajectory", agent_tools=self.tools)

        # initialize dynamic args for prompt
        self.entities = ""
        # initialize chat history
        # self.chat_history=[]
        # initiate prompt template
        template = """The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context.
          
        If the AI does not know the answer to a question, it truthfully says it does not know. 

        You are provided with information about entities the Human mentioned. If available, they are very important.

        Always check the relevant entity information before answering a question.

        Relevant entity information: {entities}

        """

        # previous conversation: 
        # {chat_history}

        # # Conversation:
        # # Human: {input}
        # # AI:

        # OPTION 1: agent = CHAT_CONVERSATIONAL_REACT_DESCRIPTION
        # initialize memory
        # self.memory = ConversationBufferMemory(llm=self.llm, memory_key="chat_history",return_messages=True, input_key="input")
        # self.memory = AgentTokenBufferMemory(memory_key="chat_history", llm=self.llm, return_messages=True, input_key="input")

        # self.chat_memory.output_key = "output"  
        # initialize CHAT_CONVERSATIONAL_REACT_DESCRIPTION agent
        self.chat_agent  = initialize_agent(self.tools, 
                                            self.llm, 
                                            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                                            # verbose=True, 
                                            memory=self.chat_memory, 
                                            return_intermediate_steps = True,
                                            handle_parsing_errors=True,
                                            callbacks = [self.handler])
        # modify default agent prompt
        prompt = self.chat_agent.agent.create_prompt(system_message=template, input_variables=['chat_history', 'input', 'entities', 'agent_scratchpad'], tools=self.tools)
        self.chat_agent.agent.llm_chain.prompt = prompt
        # switch on regular chat mode
        self.mode = "chat"


        # OPTION 2: agent = custom LLMSingleActionAgent
        # agent = self.create_custom_llm_agent()
        # self.chat_agent = AgentExecutor.from_agent_and_tools(
        #     agent=agent, tools=self.tools, verbose=True, memory=memory
        # )

        # Option 3: agent = conversational retrieval agent

        # template = f"""The following is a friendly conversation between a human and an AI. 
        # The AI is talkative and provides lots of specific details from its context.
        #   If the AI does not know the answer to a question, it truthfully says it does not know. 


        # Before answering each question, check your tools and see which ones you can use to answer the question. 

        #   Only when none can be used should you not use any tools. You should use the tools whenever you can.  


        #   You are provided with information about entities the Human mentions, if relevant.

        # Relevant entity information:
        # {self.entities}
        # """

        # self.memory = AgentTokenBufferMemory(memory_key="chat_history", llm=self.llm, return_messages=True, input_key="input")
        # system_message = SystemMessage(
        # content=(
        #     template
        #     ),
        # )
        # self.prompt = OpenAIFunctionsAgent.create_prompt(
        #     system_message=system_message,
        #     extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
        # )
        # agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        # self.chat_agent = AgentExecutor(agent=agent, tools=self.tools, memory=self.memory, verbose=True,
        #                            return_intermediate_steps=True)

        # Option 4 vectorstore agent
    
        # if (self.user_specific):
        #     router_toolkit = create_vectorstore_agent_toolkit(self.embeddings, self.llm, "specific", redis_index_name = "redis_web_advice", faiss_index_name=f"faiss_user_{self.userid}")
        #     print(f"Successfully created redis and faiss vector store toolkit")
        #     self.chat_agent = create_vectorstore_router_agent(
        #             llm=self.llm, toolkit=router_toolkit, verbose=True
        #         )
       
        # else:
        #     router_toolkit = create_vectorstore_agent_toolkit(self.embeddings, self.llm, "general", redis_index_name="redis_web_advice")
        #     print(f"Successfully created redis vector store toolkit")
        #     vs_agent = create_vectorstore_router_agent(
        #         llm=self.llm, toolkit=router_toolkit ,  verbose=True
        #     )
         
            
        

      


    # def create_custom_llm_agent(self):

    #     system_msg = "You are a helpful, polite assistant who evaluates people's resume and provides feedbacks."

    #     template = """Complete the objective as best you can. You have access to the following tools:

    #         {tools}

    #         Use the following format:

    #         Question: the input question you must answer
    #         Thought: you should always think about what to do
    #         Action: the action to take, should be one of [{tool_names}]
    #         Action Input: the input to the action
    #         Observation: the result of the action
    #         ... (this Thought/Action/Action Input/Observation can repeat N times)
    #         Thought: I now know the final answer
    #         Final Answer: the final answer to the original input question


    #         Begin!

    #         {chat_history}

    #         Question: {question}
    #         {agent_scratchpad}"""


    #     prompt = CustomPromptTemplate(
    #         template=template,
    #         tools=self.tools,
    #         system_msg=system_msg,
    #         # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    #         # This includes the `intermediate_steps` variable because that is needed
    #         input_variables=["chat_history", "question", "intermediate_steps"],
    #     )
    #     output_parser = CustomOutputParser()
    #     # LLM chain consisting of the LLM and a prompt
    #     llm_chain = LLMChain(llm=self.llm, prompt=prompt)
    #     tool_names = [tool.name for tool in self.tools]

    #     agent = LLMSingleActionAgent(
    #         llm_chain=llm_chain, 
    #         output_parser=output_parser,
    #         stop=["\nObservation:"], 
    #         allowed_tools=tool_names
    #     )

    #     return agent

    def _initialize_meta_agent(self):

        """ Initializes meta agent that will try to resolve any miscommunication between AI and Humn by providing Instruction for AI to follow.

        Args: None

        Returns: None
        
        """
 
        # It will all the logs as its tools for debugging new errors
        tools = create_search_all_chat_history_tool()
        # initiate instruct agent: ZERO_SHOT_REACT_DESCRIPTION
        prefix =  """Your job is to provide the Instructions so that AI assistant would quickly and correctly respond in the future. 
        
        Please reflect on this AI  and Human interaction below:

        ####

        {chat_history}

        ####

        If to your best knowledge AI is not correctly responding to the Human request, or if you believe there is miscommunication between Human and AI, 

        please provide a new Instruction for AI to follow so that it can satisfy the user in as few interactions as possible.

        You should format your instruction as:

        Instruction for AI: 

        If the conversation is going well, please DO NOT output any instructions and do nothing. Use your tool only if there is an error message. 
        
        """
        # Your new Instruction will be relayed to the AI assistant so that assistant's goal, which is to satisfy the user in as few interactions as possible.
        # If there is anything that can be improved about the interactions, you should revise the Instructions so that AI Assistant would quickly and correctly respond in the future.
        if self.mode == "chat":
            memory = self.chat_memory
        elif self.mode == "study":
            memory = self.study_memory

        self.meta_agent = initialize_agent(
                tools,
                self.llm, 
                agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                max_execution_time=1,
                early_stopping_method="generate",
                agent_kwargs={
                    'prefix':prefix,
                    "input_variables": ["input", "agent_scratchpad", "chat_history"]
                },
                handle_parsing_errors=True,
                memory = memory, 
                callbacks = [self.handler]
        )



    def _initialize_log(self):
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



    def _initialize_grader(self, llm = ChatOpenAI(temperature=0.0, cache=False)):

        # initialize grader agent. Grader shares the same memory as study companion
        # Conversational Retrieval Agent: https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents
        system_message = SystemMessage(
        content=(

          """You are a professional interview grader who grades the quality of responses to interview questions. 
          
          Access your memory and retrieve the very last piece of the conversation, if available.

          Determine if the AI has asked an interview question or a study material question. If it has, you are to grade the Human input based on how well it answers the question.

          You will need to use your tool "quiz_interview_material", if relevant, to search for the correct answer to the interview question.
        
          If the answer is cannot be found in the your search, use other tools or answer to your best knowledge. 

          Remember, the Human may not know the answer or may have answered the question incorrectly.

          Therefore it is important that you provide an informative feedback to the Human's response in the format:

          Feedback: <your feedback>

          Your feedback should take both the correct answer and the Human's response into consideration. When the Human's response implies that they don't know the answer, provide the correct answer in your feedback.

          If to your best knowledge the very last piece of conversation does not contain an interview question, do not provide any feedback since you only grades interview questions. 

        
            """
        )
        )
        prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
        )
        agent = OpenAIFunctionsAgent(llm=llm, tools=self.study_tools, prompt=prompt)
        self.grader_agent = AgentExecutor(agent=agent, 
                                        tools=self.study_tools, 
                                        memory=self.study_memory, 
                                        # verbose=True,
                                        return_intermediate_steps=True, 
                                        handle_parsing_errors=True,
                                        callbacks = [self.handler])
        #switch on study mode after all initializaion is done
        self.mode = "study"
        



    def _initialize_study_companion(self, json_request, llm = ChatOpenAI(temperature=0.0, cache=False)):
 
        try: 
            args = json.loads(json_request)
        except JSONDecodeError:
            args = {}       
        if "interview topic" in args:
            topic = args["interview topic"]
        else:
            topic = ""

        switch_chat_tool = self.switch_chat_tool()
        debug_tool = create_debug_tool()
        try:
            self.study_tools
        except AttributeError:
            self.study_tools = switch_chat_tool + debug_tool
        else:
            self.study_tools += switch_chat_tool + debug_tool


        #initialize qa agent
        # Conversational Retrieval Agent: https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents

        template =   f"""
            You are an AI job interviewer who asks Human interview questions and helps them understand their interview material better. 

            The specific interview content which you're to research for making personalized interview questions  is contained in the tool "quiz_interview_material", if available.

            Human may also have asked for a specific topic to study: {topic}

            If the Human is asking about other things, always answer the Human's specific question, then ask if they want to continue with the interview process.

            As an interviewer, you do not need to assess Human's response to your questions. Their response will be sent to a professional grader.         

            Sometimes you will be provided with the professional grader's feedback. You need to relay the grader's feedback to the Human in a timely manner.

            To do this, access your memory and retrieve the very last piece of conversation. If there is professional feedback in it, please always pass this on to the Human as soon as appropriate. 


            Always remember your role as an interviewer. Unless you're told to stop interviewing, you should not stop asking interview questions in the format: 

            Question: <new interview question>
   

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

        agent = OpenAIFunctionsAgent(llm=llm, tools=self.study_tools, prompt=prompt)

        # messages = chat_prompt.format_messages(
        #           grader_feedback = self.grader_feedback,
        #         instructions = self.instructions
        # )

        # prompt = OpenAIFunctionsAgent.create_prompt(
        #         # system_message=system_msg,
        #         extra_prompt_messages=messages,
        #     )
        # agent = OpenAIFunctionsAgent(llm=llm, tools=study_tools, prompt=prompt)

        self.study_agent = AgentExecutor(agent=agent,
                                    tools=self.study_tools, 
                                    memory=self.study_memory, 
                                    # verbose=True,
                                    return_intermediate_steps=True,
                                    handle_parsing_errors=True,
                                    callbacks = [self.handler])
        # call to initalize grader_agent
        self._initialize_grader()

        
    def switch_chat_agent(self, json_request) -> None:

        self.mode = "chat"
        


        
        # ask for feedback


    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff between retries
        stop=stop_after_attempt(5)  # Maximum number of retry attempts
    )
    def askAI(self, userid:str, user_input:str, callbacks=None,) -> str:

        try:    
            # BELOW IS USED WITH CONVERSATIONAL RETRIEVAL AGENT (grader_agent and study_agent)
            if (update_instruction):
                instruction = self.askMetaAgent()
            if self.mode == "study":             
                grader_feedback = self.grader_agent({"input":user_input}).get("output", "")
                print(f"GRADER FEEDBACK: {grader_feedback}")
                response = self.study_agent({"input":user_input})
                # self.study_memory.chat_memory.add_ai_message(self.instructions)
                # update chat history for instruct agent
                # self.update_chat_history(self.study_memory)
            # BELOW IS USED WITH CHAT_CONVERSATIONAL_REACT_DESCRIPTION (chat_agent)
            elif self.mode =="chat":     
                response = self.chat_agent({"input": user_input, "chat_history":[], "entities": self.entities}, callbacks = [callbacks])
                # print(f"CHAT AGENT MEMORY: {self.chat_agent.memory.buffer}")
                # update chat history for instruct agent
                # self.update_chat_history(self.chat_memory)
                # print(f"INSTRUCT AGENT MEMORY: {self.chat_agent.memory.buffer}")
                # update instruction from feedback 
                # self.update_instructions(feedback)       
            if (evaluate_result):
                evalagent_q = Queue()
                evalagent_p = Process(target = self.askEvalAgent, args=(response, evalagent_q, ))
                evalagent_p.start()
                evalagent_p.join()
                evaluation_result = evalagent_q.get()
                # add evaluation and instruction to log
                self.update_meta_data(evaluation_result)
            
            # convert dict to string for chat output
            response = response.get("output", "sorry, something happened, try again.")
        # let instruct agent handle all exceptions with feedback loop
        except Exception as e:
            print(f"ERROR HAS OCCURED IN ASKAI: {e}")
            error_msg = str(e)
            # needs to get action and action input before error and add it to error message
            if (update_instruction):
                query = f""""Debug the error message and provide Instruction for the AI assistant: {error_msg}
                    """        
                instruction = self.askMetaAgent(query)
                # self.update_instructions(feedback)
            if evaluate_result:
                self.update_meta_data(error_msg)
            error_querry = """Please debug the chat conversation"""
            self.askAI(userid, user_input, callbacks)        

        # pickle memory (sanity check)
        # with open('conv_memory/' + userid + '.pickle', 'wb') as handle:
        #     pickle.dump(self.chat_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print(f"Sucessfully pickled conversation: {chat_history}")
        return response
    

    def askMetaAgent(self, query="") -> None:    

        try: 
            feedback = self.meta_agent({"input":query}).get("output", "")
        except Exception as e:
            if type(e) == OutputParserException:
                feedback = str(e)
                feedback = feedback.removeprefix("Could not parse LLM output: `").removesuffix("`")
            else:
                raise(e)
        return feedback
    
    def askEvalAgent(self, response, queue) -> None:
        try:
            evaluation_result = self.evaluator.evaluate_agent_trajectory(
                prediction=response["output"],
                input=response["input"],
                agent_trajectory=response["intermediate_steps"],
                )
        except Exception as e:
            evaluation_result = ""
        queue.put(evaluation_result)



    # def update_chat_history(self, memory) -> None:
    #     # memory_key = chain_memory.memory_key
    #     # chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
    #     extracted_messages = memory.chat_memory.messages
    #     self.chat_history = messages_to_dict(extracted_messages)


    
    def update_study_tools(self) -> None:      

        name = f"faiss_study_material_{self.userid}"
        db = retrieve_faiss_vectorstore(self.embeddings, name)
        if db is not None:
            retriever = db.as_retriever()
            tool_name ="quiz_interview_material"
            tool_description =  """Useful for generating interview questions and answers. 
            Use this tool more than any other tool when asked to study, review, practice interview material for the topics in interview material topics. 
            Do not use this tool to load any files or documents. This should be used only for searching and generating questions and answers of the relevant topics. """
            study_tools = [create_retriever_tool(
                retriever,
                tool_name,
                tool_description
                )]
            try:
                self.study_tools
            except AttributeError:
                self.study_tools = study_tools
            else:
                self.study_tools += study_tools
            print(f"Succesfully created study companion tool: {tool_name}")
        else:
            print(f"No vectorstore for study material created yet")

        
    def update_entities(self,  text:str) -> None:
        # if "resume" in text:
        #     self.delete_entities("resume")
        # if "cover letter" in text:
        #     self.delete_entities("cover letter")
        # if "resume evaluation" in text:
        #     self.delete_entities("resume evaluation")
        self.entities += f"\n{text}\n"
        print(f"Successfully added entities {self.entities}.")

    def delete_entities(self, type: str) -> None:
        starting_idices = [m.start() for m in re.finditer(type, self.entities)]
        for idx in starting_idices:
            self.entities = self.entities.replace(self.entities[idx: idx+68], "")

    # def update_instructions(self, meta_output:str) -> None:
    #     delimiter = "Instructions: "
    #     new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
    #     self.instructions = new_instructions
    #     print(f"Successfully updated instruction to: {new_instructions}")

    def update_meta_data(self, data: str) -> None:
        with open(log_path+f"{self.userid}.log", "a") as f:
            f.write(str(data))
            print(f"Successfully updated meta data: {data}")
    

    def initiate_study_companion_tool(self) -> List[Tool]:
        
        name = "interview_preparation_helper"
        parameters = '{{"interview topic: <interview topic>"}}'
        description = f"""Initiates the study session for interview preparation.
            Use this tool whenever Human wants to practice for an interview or study interview material. 
            Also use this tool when user asks questions about a topic that's in interview material topics. This implies they want to help for the study material. 
            Input should be JSON in the following format: {parameters} 
            Output should be informing Human that you have succesfully created a study companion for them to review interview question and ask them if they could upload any study material if they have not already done so."""
        tools = [
            Tool(
            name = name,
            func = self._initialize_study_companion,
            description = description, 
            verbose = False,
            handle_tool_error=handle_tool_error,
            )
        ]
        print("Succesfully created study helper tool.")
        return tools

        

    def switch_chat_tool(self) -> List[Tool]:

        name = "interview_stopper"
        description = f"""Ends the interview or study session. Use this whenever Human asks to end the interview or study session.
                    Input is empty. 
                    Output should be asking Human to provide a brief feedback on their session experience. """
        tools = [
            Tool(
            name = name,
            func = self.switch_chat_agent,
            description=description,
            verbose=False,
            handle_tool_error=handle_tool_error,
            )
        ]
        print("Sucessfully created switch back to chat agent tool")
        return tools
    




    
    

    
        



        
        


    



    

    



    
    


    





             






    












    
    
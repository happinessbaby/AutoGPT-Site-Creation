
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
from langchain_utils import (create_vectorstore,
                             retrieve_redis_vectorstore, split_doc, CustomOutputParser, CustomPromptTemplate,
                             create_db_tools, retrieve_faiss_vectorstore, merge_faiss_vectorstore, handle_tool_error, create_search_tools, create_wiki_tools)
# from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
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
from langchain.cache import InMemoryCache



from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
log_path = os.environ["LOG_PATH"]
# debugging langchain: very useful
langchain.debug=True 
# The result evaluation process slows down chat by a lot, unless necessary, set to false
evaluate_result = False
# The instruction update process is still being tested for effectiveness
update_instruction = False
delimiter = "####"
word_count = 100
memory_max_token = 500
memory_key="chat_history"



### ERROR HANDLINGS: 
# for tools, will use custom function for all errors. 
# for chains, top will be langchain's default error handling set to True. second layer will be debub_tool with custom function for errors that cannot be automatically resolved.


        


class ChatController():

    llm = ChatOpenAI(streaming=True, model_name="gpt-3.5-turbo-0613", callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
    embeddings = OpenAIEmbeddings()
    chat_memory = ConversationBufferMemory(llm=llm, memory_key=memory_key, return_messages=True, input_key="input", output_key="output", max_token_limit=memory_max_token)
    # chat_memory = ReadOnlySharedMemory(memory=chat_memory)
    # initialize new memory (shared betweeen interviewer and grader_agent)
    interview_memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm, input_key="input", max_token_limit=memory_max_token)
    # retry_decorator = _create_retry_decorator(llm)
    langchain.llm_cache = InMemoryCache()


    def __init__(self, userid):
        self.userid = userid
        self._initialize_log()
        self._initialize_chat_agent()
        if update_instruction:
            self._initialize_meta_agent()
        


    def _initialize_chat_agent(self) -> None:

        """ Initializes main chat, a CHAT_CONVERSATIONAL_REACT_DESCRIPTION agent:  https://python.langchain.com/docs/modules/agents/agent_types/chat_conversation_agent#using-a-chat-model """
    
        # initialize tools
        # Specific purpose tools
        cover_letter_tool = create_cover_letter_generator_tool()
        resume_advice_tool = create_resume_evaluator_tool()
        interview_tool = self.initiate_interview_tool()
        # General purpose tools
        file_loader_tool = create_file_loader_tool()
        redis_store = retrieve_redis_vectorstore("index_web_advice")
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
        self.tools = general_tool + cover_letter_tool + resume_advice_tool + web_tool + interview_tool + search_tool + file_loader_tool + wiki_tool + debug_tool 


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
         
            
        

      

    def _initialize_meta_agent(self) -> None:

        """ Initializes meta agent that will try to resolve any miscommunication between AI and Humn by providing Instruction for AI to follow.  """
 
        # It will all the logs as its tools for debugging new errors
        tools = create_search_all_chat_history_tool()
        # initiate instruct agent: ZERO_SHOT_REACT_DESCRIPTION
        # prefix =  """Your job is to provide the Instructions so that AI assistant would quickly and correctly respond in the future. 
        
        # Please reflect on this AI  and Human interaction below:

        # ####

        # {chat_history}

        # ####

        # If to your best knowledge AI is not correctly responding to the Human request, or if you believe there is miscommunication between Human and AI, 

        # please provide a new Instruction for AI to follow so that it can satisfy the user in as few interactions as possible.

        # You should format your instruction as:

        # Instruction for AI: 

        # If the conversation is going well, please DO NOT output any instructions and do nothing. Use your tool only if there is an error message. 
        
        # """
        # Your new Instruction will be relayed to the AI assistant so that assistant's goal, which is to satisfy the user in as few interactions as possible.
        # If there is anything that can be improved about the interactions, you should revise the Instructions so that AI Assistant would quickly and correctly respond in the future.
        if self.mode == "chat":
            memory = ReadOnlySharedMemory(self.chat_memory)
        elif self.mode == "interview":
            memory = ReadOnlySharedMemory(self.interview_memory)

        system_msg = """You are a meta AI whose job is to provide the Instructions so that your colleague, the AI assistant, would quickly and correctly respond to Humans.
        
        Whenenver there is miscommunication between Human and your colleague, please provide a new Instruction for the AI assistant to follow so that it can satisfy the Human in as few interactions as possible."""


        template = """Complete the objective as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: Human input that can be used to assess if the conversation is going well. 
        Thought: you should always think about what to do
        Action: the action to take, should be based on Chat History below. If necessary, can be one of [{tool_names}] 
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question


        Begin!

        Chat History: {chat_history}

        Question: {question}
        {agent_scratchpad}"""


        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            system_msg=system_msg,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["chat_history", "question", "intermediate_steps"],
        )
        output_parser = CustomOutputParser()
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=self.llm, prompt=prompt, memory=memory)
        tool_names = [tool.name for tool in tools]

        self.meta_agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )



        # self.meta_agent = initialize_agent(
        #         tools,
        #         self.llm, 
        #         agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #         max_execution_time=1,
        #         early_stopping_method="generate",
        #         agent_kwargs={
        #             'prefix':prefix,
        #             "input_variables": ["input", "agent_scratchpad", "chat_history"]
        #         },
        #         handle_parsing_errors=True,
        #         memory = memory, 
        #         callbacks = [self.handler]
        # )



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



    def _initialize_interview_grader(self) -> None:


        """ Initialize interview grader agent, a Conversational Retrieval Agent: https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents """

        system_message = SystemMessage(
        content=(

          """You are a professional interview grader who grades the quality of responses to interview questions. 
          
          Access your memory and retrieve the very last piece of the conversation, if available.

          Determine if the AI has asked an interview questio. If it has, you are to grade the Human input based on how well it answers the question.

          You will need to use your tool "search_interview_material", if relevant, to search for the correct answer to the interview question.
        
          If the answer is cannot be found in the your search, use other tools or answer to your best knowledge. 

          Remember, the Human may not know the answer or may have answered the question incorrectly. Therefore it is important that you provide an informative feedback to the Human's response in the format:

          Positive Feedback: <in which ways the Human answered the question well>

          Negative Feedback: <in which ways the Human failed to answer the question>

          If to your best knowledge the very last piece of conversation does not contain an interview question, do not provide any feedback since you only grades interview questions. 

        
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
        #switch on interview mode after all initializaion is done
        self.mode = "interview"
        



    def _initialize_interview_agent(self, json_request) -> None:


        """ Initialize interviewer agent, a Conversational Retrieval Agent: https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents

        Args: 

            json_request (str): input argument from human's question, in this case the interview topic that may be contained in the question

        """
 
        try: 
            args = json.loads(json_request)
        except JSONDecodeError:
            args = {}       
        if "interview topic" in args:
            topic = args["interview topic"]
        else:
            topic = ""


        debug_tool = create_debug_tool()
        try:
            self.interview_tools
        except AttributeError:
            self.interview_tools = [self.interview_stopper] + debug_tool
        else:
            self.interview_tools += [self.interview_stopper] + debug_tool

        #initialize interviewer agent
        template =   f"""
            You are an AI job interviewer who asks Human interview questions. 

            The specific interview content which you're to research for making personalized interview questions  is contained in the tool "search_interview_material", if available.

            Human may also have asked for a specific interview topic: {topic}

            If the Human is asking about other things instead of answering an interview question, steer them back to the interview process.

            As an interviewer, you do not need to assess Human's response to your questions. Their response will be sent to a professional grader.         

            Sometimes you will be provided with the professional grader's feedback. They will be handed out to the Human at the end of the session. You should ignore them unless Huamns wants some immediate feedbacks.  

            Always remember your role as an interviewer. Unless you're told to stop interviewing, you should not stop asking interview questions in the format: 

            Question: <new interview question>

            You should ask a few personal questions at the start of the interview. Therefore, you should research the candidate's resume, cover letter, or profile too, if the relevant tools are available. 

            Sometimes you are also provided with the job position information and company information. Please do not stop asking questions until the end of the interview session. 
              

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
        # call to initalize grader_agent
        self._initialize_interview_grader()


        


        
        # ask for feedback


    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff between retries
        stop=stop_after_attempt(5)  # Maximum number of retry attempts
    )
    def askAI(self, userid:str, user_input:str, callbacks=None,) -> str:

        try:    
            # BELOW IS USED WITH CONVERSATIONAL RETRIEVAL AGENT (grader_agent and interviewer)
            if (update_instruction):
                instruction = self.askMetaAgent()
            if self.mode == "interview":             
                grader_feedback = self.grader_agent({"input":user_input}).get("output", "")
                print(f"GRADER FEEDBACK: {grader_feedback}")
                response = self.interview_agent({"input":user_input})
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

    def update_tools(self, tools, agent_type) -> None:
        if agent_type == "chat":
            self.tools += tools
        elif agent_type == "interview":
            try:
                self.interview_tools
            except AttributeError:
                self.interview_tools = tools
            else:
                self.interview_tools += tools
        print(f"Succesfully updated tool for {agent_type}")

    
    def create_interview_material_tools(self) -> List[Tool]:      

        interview_material = f"faiss_interview_material_{self.userid}"
        resume = f"faiss_resume_{self.userid}"
        cover_letter = f"faiss_cover_letter_{self.userid}"
        job_posting = f"faiss_job_posting_{self.userid}"
        vecstores = [interview_material, resume, cover_letter, job_posting]
        tools = []
        for vs in vecstores:
            db = retrieve_faiss_vectorstore(self.embeddings, vs)
            if db is not None:
                retriever = db.as_retriever()
                name = vs.removeprefix("faiss_").removesuffix(f"_{self.userid}")
                tool_name =f"search_{name}"
                tool_description =  """Useful for generating interview questions and answers. 
                Use this tool more than any other tool during a mock interview session. This tool can also be used for questions about topics in the interview material topics. 
                Do not use this tool to load any files or documents. This should be used only for searching and generating questions and answers of the relevant materials and topics. """
                tool = [create_retriever_tool(
                    retriever,
                    tool_name,
                    tool_description
                    )]
                print(f"Succesfully created interview material tool: {tool_name}")
                tools+= tool
            else:
                print(f"No vectorstore for {vs} created yet")
        return tools


        
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
        file_name_len = 68
        for idx in starting_idices:
            self.entities = self.entities.replace(self.entities[idx: idx+file_name_len], "")

    # def update_instructions(self, meta_output:str) -> None:
    #     delimiter = "Instructions: "
    #     new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
    #     self.instructions = new_instructions
    #     print(f"Successfully updated instruction to: {new_instructions}")

    def update_meta_data(self, data: str) -> None:
        with open(log_path+f"{self.userid}.log", "a") as f:
            f.write(str(data))
            print(f"Successfully updated meta data: {data}")
    

    def initiate_interview_tool(self) -> List[Tool]:
        
        name = "interview_initiator"
        parameters = '{{"interview topic: <interview topic>"}}'
        description = f"""Initiates the interview process.
            Use this tool whenever Human wants to conduct a mock interview. Do not use this tool for study purposes or answering interview questions. 
            Input should be JSON in the following format: {parameters} 
            Output should be informing Human that you have succesfully created an AI interviewer and ask them if they could upload any interview material if they have not already done so."""
        tools = [
            Tool(
            name = name,
            func = self._initialize_interview_agent,
            description = description, 
            verbose = False,
            handle_tool_error=handle_tool_error,
            )
        ]
        print("Succesfully created interview initiator tool.")
        return tools

    @tool
    def interview_stopper(self, query) -> str:
        """Ends the interview session. Use this whenever Human asks to end the interview session.
            Input is empty. """        
        self.mode = "chat"
        return "Please provide a brief feedback on your experience"




    
    

    
        



        
        


    



    

    



    
    


    





             






    












    
    
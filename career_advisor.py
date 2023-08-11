
import os
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, LLMChain
# from langchain.agents import ConversationalChatAgent, Tool, AgentExecutor
# from basic_utils import read_txt
from langchain_utils import (create_QASource_chain, create_qa_tools, create_vectorstore, 
                             retrieve_redis_vectorstore, split_doc, CustomOutputParser, CustomPromptTemplate,
                             create_db_tools, retrieve_faiss_vectorstore, merge_faiss_vectorstore)
# from langchain.prompts import BaseChatPromptTemplate
# from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain.schema import AgentAction, AgentFinish, HumanMessage
# from typing import List, Union
# import re
# from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict, AgentAction
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.docstore import InMemoryDocstore
from langchain.tools.human.tool import HumanInputRun
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import get_openai_callback, StdOutCallbackHandler, FileCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
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
from generate_cover_letter import create_cover_letter_generator_tool
from upgrade_resume import create_resume_evaluator_tool
# from feast import FeatureStore
import pickle
import json
import langchain
import faiss
from loguru import logger
from langchain.evaluation import load_evaluator
from basic_utils import convert_to_txt
from langchain.schema import OutputParserException
from multiprocessing import Process, Queue, Value


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# debugging log: very useful
langchain.debug=True
evaluate_result = True


delimiter = "####"
advice_path = "./static/advice/"
resume_path = "./uploads/"
log_path = "./log/"


class ChatController(object):


    def __init__(self, userid):
        self.userid = userid
        self.llm = ChatOpenAI(temperature=0.5, model_name="gpt-4", cache=False, streaming=True,)
        self.embeddings = OpenAIEmbeddings()
        self.handler = None
        self._initialize_chat_agent()
        self._initialize_meta_agent()
        self._initialize_log()


    def _initialize_chat_agent(self):
    

        # initialize tools
        cover_letter_tool = create_cover_letter_generator_tool()

        resume_advice_tool = create_resume_evaluator_tool()
        
        redis_store = retrieve_redis_vectorstore(self.embeddings, "index_web_advice")
        redis_retriever = redis_store.as_retriever()
        general_tool_description = """This is a general purpose database. Use it to answer general job related questions. 
        Prioritize other tools over this tool. """
        general_tool= create_db_tools(self.llm, redis_retriever, "redis_general", general_tool_description)

        search = GoogleSearchAPIWrapper()
        embedding_size = 1536  
        index = faiss.IndexFlatL2(embedding_size)  
        vectorstore = FAISS(self.embeddings.embed_query, index, InMemoryDocstore({}), {})
        web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=vectorstore,
            llm=self.llm, 
            search=search, 
        )
        web_tool_description="""This is a web research tool. Use it only when you cannot find answers elsewhere. Always add source information."""
        web_tool = create_db_tools(self.llm, web_research_retriever, "faiss_web", web_tool_description)

        self.tools = general_tool + cover_letter_tool + resume_advice_tool + web_tool 

        # initialize callback handler and evaluator
        if (evaluate_result):
            logfile = log_path + f"{self.userid}.log"
            self.handler = FileCallbackHandler(logfile)
            logger.add(logfile,  enqueue=True)
            self.evaluator = load_evaluator("trajectory", agent_tools=self.tools)


        # initialize dynamic args for tools
        self.entities = ""

        # initialize feedbacks
        self.instructions = ""

        template = """The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context.
          
        If the AI does not know the answer to a question, it truthfully says it does not know. 


        You are provided with information about entities the Human mentions, if relevant.

        Relevant entity information: {entities}

        Instructions: {instructions}

        """

        # previous conversation: 
        # {chat_history}

        # # Conversation:
        # # Human: {input}
        # # AI:

        # prompt = CustomPromptTemplate(
        #     template=self.template,
        #     tools=self.tools,
        #     # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        #     # This includes the `intermediate_steps` variable because that is needed
        #     input_variables=["chat_history", "input", "intermediate_steps", "entities"],
        # )

        # OPTION 1: agent = CHAT_CONVERSATIONAL_REACT_DESCRIPTION
        # initialize memory
        # self.memory = ConversationBufferMemory(llm=self.llm, memory_key="chat_history",return_messages=True, input_key="input")
        # self.memory = AgentTokenBufferMemory(memory_key="chat_history", llm=self.llm, return_messages=True, input_key="input")
        self.memory = ConversationBufferMemory(llm=self.llm, memory_key="chat_history", return_messages=True, input_key="input")
        self.memory.output_key = "output"
        
        # initialize agent
        self.chat_agent  = initialize_agent(self.tools, 
                                            self.llm, 
                                            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                                            # verbose=True, 
                                            memory=self.memory, 
                                            return_intermediate_steps = True,
                                            handle_parsing_errors=True,
                                            callbacks = [self.handler])
        # modify default agent prompt
        prompt = self.chat_agent.agent.create_prompt(system_message=template, input_variables=['chat_history', 'input', 'entities', 'agent_scratchpad', 'instructions'], tools=self.tools)
        self.chat_agent.agent.llm_chain.prompt = prompt



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

        db = create_vectorstore(self.embeddings, "faiss", "./log/", "dir", "chat_debug")
        description = """This is used for debugging chat errors when you encounter one. 
        This stores all the chat history in the past. """
        tools = create_db_tools(self.llm, db.as_retriever(), "chat_debug", description)
        meta_template = """
        Assistant has just had the below interactions with a User. Assistant followed their "Instructions" closely. Your job is to revise the Instructions so that Assistant would quickly and correctly respond in the future.

        ####

        {chat_history}

        ####


        Please reflect on these interactions. Use tools, if needed, to reference old instructions and improve upon them/
        
        If there's a error message, treat it as your priority. 

        You should revise the Instructions so that Assistant would quickly and correctly respond in the future. Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions. Don't forget any important details in the current Instructions! Indicate the new Instructions by "Instructions: ...".
        """
        self.meta_agent = initialize_agent(
                tools,
                self.llm, 
                agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                 agent_kwargs={
                    'prefix':meta_template,
                    # 'format_instructions':FORMAT_INSTRUCTIONS,
                    # 'suffix':SUFFIX
                    "input_variables": ["input", "agent_scratchpad","chat_history"]
                },
        )



    def _initialize_log(self):
        for path in  Path("./log/").glob('**/*.log'):
            file = str(path)
            file_name = path.stem
            if file_name is not self.userid: 
                # convert all non-empty log from previous sessions to txt and delete the log
                if os.stat(file).st_size != 0:  
                    convert_to_txt(file, f"./log/{file_name}.txt")
                os.remove(file)



    def askAI(self, userid, question, callbacks=None):

        try:
            # print("Successfully reinitialized agent")            
            # BELOW IS USED WITH CHAT_CONVERSATIONAL_REACT_DESCRIPTION 
            response = self.chat_agent({"input": question, "chat_history":[], "entities": self.entities, "instructions":self.instructions}, callbacks = [callbacks])
            # update instruction from feedback 
            self.update_chat_history()
            p = Process(target = self.askMetaAgent, args=())
            p.start()

            if (evaluate_result):
                try:
                    evaluation_result = self.evaluator.evaluate_agent_trajectory(
                        prediction=response["output"],
                        input=response["input"],
                        agent_trajectory=response["intermediate_steps"],
                        )
                except Exception as e:
                    evaluation_result = ""
                # print(evaluation_result)
                # add evaluation and instruction to log
                self.update_meta_data(evaluation_result)

            p.join()         
            # convert dict to string for chat output
            response = response.get("output", "sorry, something happened, try again.")
        # let meta agent handle all exceptions with feedback loop
        except Exception as e:
            # needs to get action and action input before error and add it to error message
            error_msg = str(e)
            query = f""""Debug the error message and provide instruction for Assistent for the next step: {error_msg}
                """
            p = Process(target = self.askMetaAgent, args=(query))
            p.start()
            self.update_meta_data(error_msg)
            p.join()
            self.askAI(userid, question, callbacks)        


        # pickle memory (sanity check)
        with open('conv_memory/' + userid + '.pickle', 'wb') as handle:
            pickle.dump(self.chat_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print(f"Sucessfully pickled conversation: {chat_history}")

        return response
    

    def askMetaAgent(self, query = ""):    
        try: 
            feedback = self.meta_agent({"input":query, "chat_history":self.chat_history}).get("output", "")
        except Exception as e:
            if type(e) == OutputParserException:
                feedback = str(e)
                feedback = feedback.removeprefix("Could not parse LLM output: `").removesuffix("`")
            else:
                raise(e)
        self.update_instructions(feedback)

    def update_chat_history(self):
        # memory_key = chain_memory.memory_key
        # chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
        extracted_messages = self.memory.chat_memory.messages
        self.chat_history = messages_to_dict(extracted_messages)
        # print(chat_history)


    
    def add_tools(self, userid, tool_name, tool_description):      
        try:
            faiss_store = retrieve_faiss_vectorstore(self.embeddings, f"{tool_name}_{userid}")
            faiss_retriever = faiss_store.as_retriever()
            specific_tool = create_db_tools(self.llm, faiss_retriever, tool_name, tool_description)
            self.tools += specific_tool
            print(f"Successfully added tool {specific_tool}")
        except Exception as e:
            raise e
        
    def update_entities(self, userid, text):
        self.entities += f"\n{text}\n"
        print(f"Successfully added {self.entities}.")


    def update_instructions(self, meta_output):
        delimiter = "Instructions: "
        new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
        self.instructions = new_instructions
        print(f"Successfully updated instruction to: {new_instructions}")

    def update_meta_data(self, data):
         with open(log_path+f"{self.userid}.log", "a") as f:
             f.write(str(data))
    





             






    












    
    
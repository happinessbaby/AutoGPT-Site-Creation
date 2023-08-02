
import os
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate
# from langchain.agents import ConversationalChatAgent, Tool, AgentExecutor
# from basic_utils import read_txt
from langchain_utils import (create_QA_chain, create_QASource_chain, create_qa_tools, create_doc_tools, create_search_tools, 
                             retrieve_redis_vectorstore, split_doc, CustomOutputParser, CustomPromptTemplate,
                             create_db_tools, retrieve_faiss_vectorstore)
# from langchain.prompts import BaseChatPromptTemplate
# from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain.schema import AgentAction, AgentFinish, HumanMessage
# from typing import List, Union
# import re
# from langchain import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
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


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

delimiter = "####"

advice_path = "./static/advice/"
resume_path = "./uploads/"
resume_uploaded=False



class ChatController(object):


    def __init__(self, userid):
        self.userid = userid
        self.llm = ChatOpenAI(temperature=0.5, model_name="gpt-4", cache=False)
        self.embeddings = OpenAIEmbeddings()
        # self.tools = []
        # self.PROMPT = None
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, memory_key="chat_history", max_token_limit=2000, return_messages=True, input_key="input")
        self._create_chat_agent()

    def _create_chat_agent(self):
    
        # # OPTION 1: agent = CHAT_CONVERSATIONAL_REACT_DESCRIPTION
        cover_letter_tool = create_cover_letter_generator_tool()

        resume_advice_tool = create_resume_evaluator_tool()
        
        redis_store = retrieve_redis_vectorstore(self.embeddings, "index_web_advice")
        redis_retriever = redis_store.as_retriever()
        general_tool_description = """This is a general purpose database. Use it to answer general job related questions. 
        Prioritize other tools over this tool. """
        general_tool= create_db_tools(self.llm, redis_retriever, "redis_general", general_tool_description)

        self.tools = general_tool + cover_letter_tool + resume_advice_tool

        # if (retrieve_faiss_vectorstore(self.embedding, f"faiss_user_{self.userid}"))!=None:
        #     faiss_store = retrieve_faiss_vectorstore(self.embeddings, f"faiss_user_{self.userid}")
        #     faiss_retriever = faiss_store.as_retriever()
        #     specific_tool = create_db_tools(self.llm, faiss_retriever, "faiss_specific")
        #     self.tools +=  specific_tool

        # TODO: 
        #  it is capable of delegating tasks to experts to complete and get reports back
        # Therefore, think about how to improve the instructions. 
        template = """The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context.
          If the AI does not know the answer to a question, it truthfully says it does not know. 

        Summary of conversaion:
        {chat_history}

        Conversation:
        Human: {input}
        AI:"""
        #   You are provided with information about entities the Human mentions, if relevant.

        # Relevant entity information:
        # {entities}
        self.PROMPT = PromptTemplate(input_variables=["chat_history", "input"], template=template)  


        self.chat_agent  = initialize_agent(self.tools, self.llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=self.memory, prompt=self.PROMPT,   handle_parsing_errors=True,)

    
    
        # OPTION 2: agent = custom LLMSingleActionAgent
        # agent = self.create_custom_llm_agent()
        # self.chat_agent = AgentExecutor.from_agent_and_tools(
        #     agent=agent, tools=self.tools, verbose=True, memory=memory
        # )

        # Option 3: agent = Plant & Execute
        # planner = load_chat_planner(self.llm)
        # executor = load_agent_executor(self.llm, self.tools, verbose=True)
        # self.chat_agent = PlanAndExecute(planner=planner, executor=executor, verbose=True, memory=memory)

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

    def add_tools(self, tool_name, tool_description):       
        try:
            faiss_store = retrieve_faiss_vectorstore(self.embeddings, f"faiss_user_{self.userid}")
            faiss_retriever = faiss_store.as_retriever()
            specific_tool = create_db_tools(self.llm, faiss_retriever, tool_name, tool_description)
            self.tools +=  specific_tool
        except Exception as e:
            raise e

    def update_prompt():
        return None



    def askAI(self, userid, question, callbacks=None):

        # retrieve a conversation memory 
        # can be changed to/incorporated into a streaming platform such as kafka
        if os.path.isfile('./conv_memory/'+userid+'.pickle'):
            # retrieve pickled memory
            with open('./conv_memory/'+userid+'.pickle', 'rb') as handle:
                serializable_mem = pickle.load(handle)
                print(f"Sucessfully loaded pickled conversation: {serializable_mem}")
            retrieved_messages = messages_from_dict(serializable_mem) 
            self.chat_history= ChatMessageHistory(messages=retrieved_messages) 
            self.memory = ConversationSummaryBufferMemory(llm=self.llm, memory_key="chat_history", chat_memory=self.chat_history, max_token_limit=2000, return_messages=True, input_key="input")
            self.chat_agent  = initialize_agent(self.tools, self.llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=self.memory, prompt=self.PROMPT,  handle_parsing_errors=True,)
            print("Succesfully updated chat agent memory")
        
        # for could not parse LLM output
        # print(f"Chat history: {self.chat_history}")
        try:
            # response = self.chat_agent.run(input=question)
            # response = self.chat_agent.run(question, callbacks=callbacks)
            # BELOW IS USED WITH CHAT_CONVERSATIONAL_REACT_DESCRIPTION 
            response = self.chat_agent({"input": question, "chat_history": []}, callbacks=callbacks)
                
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                print(e)
                raise e
            response = response.removeprefix(
                "Could not parse LLM output: `").removesuffix("`")

        
        # save memory after response
        extracted_messages = self.memory.chat_memory.messages
        serializable_mem = messages_to_dict(extracted_messages)
        # pickle memory
        with open('conv_memory/' + userid + '.pickle', 'wb') as handle:
            pickle.dump(serializable_mem, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Sucessfully pickled conversation: {serializable_mem}")


        return response
    
    
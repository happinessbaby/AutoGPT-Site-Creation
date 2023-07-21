
import os
from openai_api import evaluate_response
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from langchain.agents import ConversationalChatAgent, Tool, AgentExecutor
from langchain_utils import create_QA_chain, create_qa_tools, create_doc_tools, CustomOutputParser, CustomPromptTemplate
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Union
import re
from langchain import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
# from feast import FeatureStore
import pickle
# from fastapi import HTTPException
# from bson import ObjectId

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

llm = ChatOpenAI(temperature=0.0)
# llm = OpenAI(temperature=0, top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)
embeddings = OpenAIEmbeddings()

class ChatController(object):


    def __init__(self, advice_file):
        # self.advice = read_txt(advice_file)
        self.advice_file = advice_file
        self.llm = OpenAI(temperature=0, model_name="gpt-4", top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)
        self.tools = []
        self._create_chat_agent()


    def _create_chat_agent(self):


        qa = create_QA_chain(self.llm, embeddings, "chroma" )

        self.tools = create_qa_tools(qa)

        self.tools += create_doc_tools(self.advice_file)

        # self.tools += create_process_tools("generate_cover_letter.py")

        # system_msg = "You are a helpful assistant who evaluates a human's resume and provides resume advices."

        # prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools: 
        
        # {tools}
        
        # An initial assessment of the resume has been done. The assessment is is delimited with {delimiter} characters. 
        
        # Use it as a reference when answering questions.
        
        # """
        # suffix = """Begin!"

        # {chat_history}
        # Question: {input}
        # {agent_scratchpad}"""

        # # This probably can be changed to Custom Agent class
        # agent = ConversationalChatAgent.from_llm_and_tools(
        #     llm=self.llm,
        #     tools=tools,
        #     system_message=system_msg,
        #     prefix=prefix,
        #     suffix= suffix,
        # )

            # Set up the base template
    
        agent = self.create_custom_llm_agent()
        
        memory = ConversationSummaryBufferMemory(llm=self.llm, memory_key="chat_history", max_token_limit=650, return_messages=True, input_key="question")

        self.chat_agent = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self.tools, verbose=True, memory=memory
        )


    def create_custom_llm_agent(self):

        system_msg = "You are a helpful, polite, and mindful assistant who evaluates people's resume and provides feedbacks."

        template = """Complete the objective as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question


            Begin!

            {chat_history}

            Question: {question}
            {agent_scratchpad}"""


        prompt = CustomPromptTemplate(
            template=template,
            tools=self.tools,
            system_msg=system_msg,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["chat_history", "question", "intermediate_steps"],
        )
        output_parser = CustomOutputParser()
        # LLM chain consisting of the LLM and a prompt
        # memory = ConversationBufferMemory(memory_key="chat_history", k=50, return_messages=True, input_key="question" )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in self.tools]

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )

        return agent


    def askAI(self, id, question, callbacks=None):
        # try:
        #     objId = ObjectId(id)
        # except:
        #     raise HTTPException(status_code=400, detail="Not valid id.")

        # create a conversation memory and save it if it not exists 
        # can be changed to/incorporated into a streaming platform such as kafka
        # if not os.path.isfile('conv_memory/'+id+'.pickle'):
        #     mem = ConversationSummaryBufferMemory(llm=self.llm, memory_key="chat_history", input_key="question",return_messages=True)
        #     with open('conv_memory/' + id + '.pickle', 'wb') as handle:
        #         pickle.dump(mem, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # else:
        #     # load the memory according to the user id
        #     with open('conv_memory/'+id+'.pickle', 'rb') as handle:
        #         mem = pickle.load(handle)

        # self.chat_agent.memory = mem

        # for could not parse LLM output
        try:
            # response = self.chat_agent.run(input=prompt)
            response = self.chat_agent.run(question, callbacks=callbacks)
            # response = self.chat_agent({"question": question, "delimiter":delimiter, "assessment":self.advice}, callbacks=callbacks)
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                print(e)
                raise e
            response = response.removeprefix(
                "Could not parse LLM output: `").removesuffix("`")

        # save memory after response
        # with open('conv_memory/' + id + '.pickle', 'wb') as handle:
        #     pickle.dump(mem, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return response
    
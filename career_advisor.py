
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
from langchain.agents import AgentType, Tool
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict, AgentAction
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.docstore import InMemoryDocstore
from langchain.agents import AgentExecutor
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
from common_utils import create_file_loader_tool
from langchain.tools import tool, format_tool_to_openai_function
import re



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




        


class ChatController():

     

    def __init__(self, userid):
        self.userid = userid
        self.llm = ChatOpenAI(temperature=0.5, model_name="gpt-4", cache=False, streaming=True,)
        self.embeddings = OpenAIEmbeddings()
        self.handler = None
        self._initialize_chat_agent()
        if update_instruction:
            self._initialize_instruct_agent()
        self._initialize_log()
        # self._initialize_study_companion()


    def _initialize_chat_agent(self):
    
        # initialize tools
        # Specific purpose tools
        cover_letter_tool = create_cover_letter_generator_tool()
        resume_advice_tool = create_resume_evaluator_tool()
        file_loader_tool = create_file_loader_tool()
        study_companion_tool = self.initiate_study_companion_tool()
        # General purpose tools
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
        self.web_tool = create_db_tools(self.llm, web_research_retriever, "faiss_web", web_tool_description)
        # gather all the tools together
        self.tools = general_tool + cover_letter_tool + resume_advice_tool + self.web_tool + file_loader_tool + study_companion_tool

        # initialize file callback logging
        logfile = log_path + f"{self.userid}.log"
        self.handler = FileCallbackHandler(logfile)
        logger.add(logfile,  enqueue=True)

        # initialize evaluator
        if (evaluate_result):
            self.evaluator = load_evaluator("trajectory", agent_tools=self.tools)

        # initialize dynamic args for prompt
        self.entities = ""
        # initialize feedbacks
        self.instructions = ""
        # initialize chat history
        self.chat_history=[]
        # initiate prompt template
        template = """The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context.
          
        If the AI does not know the answer to a question, it truthfully says it does not know. 

        You are provided with information about entities the Human mentions. If available, they are very important.

        Always check the relevant entity information before answering a question.

        Relevant entity information: {entities}

        You are also provided with specific questions that would help the Human. If relevant, ask them.

        Instructions: {instructions}

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
        self.chat_memory = ConversationBufferMemory(llm=self.llm, memory_key="chat_history", return_messages=True, input_key="input")
        self.chat_memory.output_key = "output"  
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
        prompt = self.chat_agent.agent.create_prompt(system_message=template, input_variables=['chat_history', 'input', 'entities', 'agent_scratchpad', 'instructions'], tools=self.tools)
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

    def _initialize_instruct_agent(self):
        # This is an instruction agent that provides main agent with improved instructions, also iused for debugging errors during chat
        # It will all the logs as its tools for debugging new errors
        db = create_vectorstore(self.embeddings, "faiss", "./log/", "dir", "chat_debug")
        # TODO: need to test the effective of this debugging tool
        name = "search_all_chat_history"
        description = """Useful when you want to revise instructions based on historical chat conversations. Use it especially when debugging chat error messages. """
        tools = create_db_tools(self.llm, db.as_retriever(), name, description)
        meta_template = """
        Assistant has just had the below interactions with a User. Assistant followed their "Instructions" closely. Your job is to revise the Instructions so that Assistant would quickly and correctly respond in the future.

        ####

        {chat_history}

        ####


        Please reflect on these interactions. Use tools, if needed, to reference old instructions and improve upon them/
        
        If there's a error message, treat it as your priority. 

        You should revise the Instructions so that Assistant would quickly and correctly respond in the future. Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions. Don't forget any important details in the current Instructions! Indicate the new Instructions by "Instructions: ...".
        """
        # initiate instruct agent: ZERO_SHOT_REACT_DESCRIPTION
        self.instruct_agent = initialize_agent(
                tools,
                self.llm, 
                agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                max_execution_time=1,
                early_stopping_method="generate",
                agent_kwargs={
                    'prefix':meta_template,
                    # 'format_instructions':FORMAT_INSTRUCTIONS,
                    # 'suffix':SUFFIX
                    "input_variables": ["input", "agent_scratchpad","chat_history"]
                },
        )



    def _initialize_log(self):
        # Upon start, all the .log files will be deleted and changed to .txt files
        for path in  Path(log_path).glob('**/*.log'):
            file = str(path)
            file_name = path.stem
            if file_name != self.userid: 
                # convert all non-empty log from previous sessions to txt and delete the log
                if os.stat(file).st_size != 0:  
                    convert_to_txt(file, f"./log/{file_name}.txt")
                os.remove(file)



    def _initialize_grader(self, embeddings=OpenAIEmbeddings(), llm = ChatOpenAI(temperature=0.0, cache=False)):

        # initialize grader agent. Grader shares the same memory as study companion
        # Conversational Retrieval Agent: https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents
        system_message = SystemMessage(
        content=(
          """Access your memory and retrieve the very last piece of the conversation, if available.

          Determine if the AI has asked an interview question. If it has, you are to grade the Human input based on how well it answers the interview question.

          You will need to use your tool "quiz_interview_material", if relevant, to generate the correct answer to the interview question.

          The Human may not know the answer or answered the question incorrectly.

          Therefore it is important that you provide an informative feedback to the Human in the format:

          Feedback: <your feedback>

          If to your best knowledge there was no interview question, or the Human is not answering in any way, you should not provide any feedback.
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
            topic = "none"

        try:
            self.study_tools
        except AttributeError:
            self.study_tools = self.web_tool

        # initialize new memory (shared betweeen study_agent and grader_agent)
        self.study_memory = AgentTokenBufferMemory(memory_key="chat_history", llm=llm, input_key="input")

        #initialize qa agent
        # Conversational Retrieval Agent: https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents

        template =   f"""
            You are an AI job interviewer. Your role is to ask Human interview questions and help them understand their interview material better. 

            The main interview content which you're to research for making the interview question  is contained in the tool "quiz_interview_material", if available.

            The main topic of the interview is: {topic} \n

            If the Human is asking a specific question instead, always answer the Human's specific question, then ask if they want to continue with the interview process.

            During  the interview process, if the Human is not asking a specific question, then the Human is answering an interview question.

            You can ignore the answer as it will be sent to a professional grader. 

            Sometimes you will be provided with the grader's feedback. You need to relay the grader's feedback to the Human.

            To do this, access your memory and retrieve the last piece of conversation. If there is Feedback in it, please always pass this on to the Human. 

            Remember, unless you're told to stop, you should not stop asking interview questions in the format: 

            Question: <new interview question>

            Also remember, do not ask the same question twice!

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
        




    def askAI(self, userid:str, user_input:str, callbacks=None,) -> str:

        try:    
            # BELOW IS USED WITH CONVERSATIONAL RETRIEVAL AGENT (grader_agent and study_agent)
            if self.mode == "study":             
                grader_feedback = self.grader_agent({"input":user_input}).get("output", "")
                print(f"GRADER FEEDBACK: {grader_feedback}")
                response = self.study_agent({"input":user_input})
                # update chat history for instruct agent
                self.update_chat_history(self.study_memory)
            # BELOW IS USED WITH CHAT_CONVERSATIONAL_REACT_DESCRIPTION (chat_agent)
            elif self.mode =="chat":     
                response = self.chat_agent({"input": user_input, "chat_history":[], "entities": self.entities , "instructions":self.instructions}, callbacks = [callbacks])
                # update chat history for instruct agent
                self.update_chat_history(self.chat_memory)
            if (update_instruction):
                instructagent_q = Queue()
                instructagent_p = Process(target = self.askInstructAgent, args=(instructagent_q, ))
                instructagent_p.start()   
                instructagent_p.join() 
                feedback = instructagent_q.get()
                # update instruction from feedback 
                self.update_instructions(feedback)       
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
            print(e)
            # needs to get action and action input before error and add it to error message
            if (update_instruction):
                error_msg = str(e)
                query = f""""Debug the error message and provide instruction for Assistent for the next step: {error_msg}
                    """
                instructagent_q = Queue()
                instructagent_p = Process(target = self.askInstructAgent, args=(instructagent_q, query, ))
                instructagent_p.start()
                instructagent_p.join()
                feedback = instructagent_q.get()
                self.update_instructions(feedback)
            if evaluate_result:
                self.update_meta_data(error_msg)
           
            self.askAI(userid, user_input, callbacks)        

        # pickle memory (sanity check)
        with open('conv_memory/' + userid + '.pickle', 'wb') as handle:
            pickle.dump(self.chat_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print(f"Sucessfully pickled conversation: {chat_history}")
        return response
    

    def askInstructAgent(self, queue, query = "") -> None:    

        try: 
            feedback = self.instruct_agent({"input":query, "chat_history":self.chat_history}).get("output", "")
        except Exception as e:
            if type(e) == OutputParserException:
                feedback = str(e)
                feedback = feedback.removeprefix("Could not parse LLM output: `").removesuffix("`")
            else:
                raise(e)
        queue.put(feedback)
    
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



    def update_chat_history(self, memory) -> None:
        # memory_key = chain_memory.memory_key
        # chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
        extracted_messages = memory.chat_memory.messages
        self.chat_history = messages_to_dict(extracted_messages)


    
    def update_study_tools(self) -> None:      

        name = f"faiss_study_material_{self.userid}"
        db = retrieve_faiss_vectorstore(self.embeddings, name)
        if db is not None:
            retriever = db.as_retriever()
            tool_name ="quiz_interview_material"
            tool_description =  """Useful for generating interview questions and answers.
            Use this tool more than any other tool when asked to study, review, practice interview material for the given content."""
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
            print(f"Succesfully created study companion tool: {study_tools}")
        else:
            print(f"Failed to initalize study tool for name: {name}")

        
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

    def update_instructions(self, meta_output:str) -> None:
        delimiter = "Instructions: "
        new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
        self.instructions = new_instructions
        print(f"Successfully updated instruction to: {new_instructions}")

    def update_meta_data(self, data: str) -> None:
        with open(log_path+f"{self.userid}.log", "a") as f:
            f.write(str(data))
            print(f"Successfully updated meta data: {data}")
    

    def initiate_study_companion_tool(self) -> List[Tool]:
        
        name = "interview_preparation_helper"
        parameters = "interview topic: <interview topic>"
        description = f"""Initiates the interview preparation process.  
            Use this tool whenever Human wants to practice for an interview or study interview material. 
            Input should be JSON in the following format: {parameters}
            Output should be informing Human that you have succesfully created a study companion for them to review interview question and ask them if they could upload any study material if they have not already done so."""
        tools = [
            Tool(
            name = name,
            func = self._initialize_study_companion,
            description = description, 
            verbose = False,
            )
        ]
        print("Succesfully created study helper tool.")
        return tools

    
        



        
        


    



    

    



    
    


    





             






    












    
    
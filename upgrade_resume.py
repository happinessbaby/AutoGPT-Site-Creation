import os
from openai_api import evaluate_response
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from basic_utils import read_txt, check_content_safety
from common_utils import fetch_samples, get_web_resources
from samples import resume_samples_dict
from langchain.agents import ConversationalChatAgent, Tool, AgentExecutor
from langchain_utils import create_QA_chain, create_qa_tools, CustomOutputParser, CustomPromptTemplate
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
# randomize delimiters from a delimiters list to prevent prompt injection
delimiter = "####"
delimiter1 = "````"
delimiter2 = "////"
delimiter3 = "<<<<"
delimiter4 = "****"

my_job_title = 'AI developer'
my_resume_file = 'resume_samples/resume2023v2.txt'

class ChatController(object):


    def __init__(self, advice_file):
        self.advice = read_txt(advice_file)
        self.llm = OpenAI(temperature=0, model_name="gpt-4", top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)
        self._create_chat_agent()


    def _create_chat_agent(self):


        qa = create_QA_chain(self.llm, embeddings, "chroma" )

        tools = create_qa_tools(qa)

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
    
        agent = self.create_custom_llm_agent(tools)
        
        memory = ConversationSummaryBufferMemory(llm=self.llm, memory_key="chat_history", max_token_limit=650, return_messages=True, input_key="question")

        self.chat_agent = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )


    def create_custom_llm_agent(self, tools):

        system_msg = "You are a helpful assistant who evaluates a human's resume and provides resume advices."

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

            An initial assessment of the resume has been done. The assessment is delimited with {delimiter} characters.

            Use it as a reference when answering questions.

            assessment: {delimiter}{assessment}{delimiter}

            Begin!

            {chat_history}

            Question: {question}
            {agent_scratchpad}"""

        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            system_msg=system_msg,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["delimiter", "assessment", "chat_history", "question", "intermediate_steps"],
        )
        output_parser = CustomOutputParser()
        # LLM chain consisting of the LLM and a prompt
        # memory = ConversationBufferMemory(memory_key="chat_history", k=50, return_messages=True, input_key="question" )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]

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
        if not os.path.isfile('conv_memory/'+id+'.pickle'):
            mem = ConversationSummaryBufferMemory(llm=self.llm, memory_key="chat_history", input_key="question",return_messages=True)
            with open('conv_memory/' + id + '.pickle', 'wb') as handle:
                pickle.dump(mem, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # load the memory according to the user id
            with open('conv_memory/'+id+'.pickle', 'rb') as handle:
                mem = pickle.load(handle)

        self.chat_agent.memory = mem

        # for could not parse LLM output
        try:
            # response = self.chat_agent.run(input=prompt)
            # response = self.chat_agent.run(question, callbacks=callbacks)
            response = self.chat_agent({"question": question, "delimiter":delimiter, "assessment":self.advice}, callbacks=callbacks)
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                print(e)
                raise e
            response = response.removeprefix(
                "Could not parse LLM output: `").removesuffix("`")

        # save memory after response
        with open('conv_memory/' + id + '.pickle', 'wb') as handle:
            pickle.dump(mem, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return {"answer": response}
    


    

def basic_upgrade_resume(my_job_title, read_path = my_resume_file, res_path="advice.txt"):

    resume = read_txt(read_path)
    query  = f"""Find out what a {my_job_title} does and the skills and responsibilities involved. """
    job_description = get_web_resources(llm, query)
    resume_advices = get_web_resources(llm, "what makes a bad resume and how to improve")
    resume_samples = fetch_samples(llm, embeddings, my_job_title, resume_samples_dict)

    template_string = """" Your task is to analyze the weaknesses of a resume and ways to improve it. 
        
    The resume is delimited with {delimiter1} chararacters.
    
    resume: {delimiter1}{resume}{delimiter1}

    Step 1: Search and extract fields of the resume.  

         Some common resume fields include but not limited to personal information, objective, education, work experience, awards and honors, and skills.

    Step 2: Determine if information contained in the fields such as objective, work experience and skills is relevant to {job} and {job}'s job description.

        Generate a list of irrelevant information that should not be included in the resume for {job}. 
     
      job description: {delimiter4}{job_description}{delimiter4}. 

    Step 3: Research sample resume provided. Each sample is delimited with {delimiter2} characters.

        Compare them with resume in Step 1 and list out contents that are in the samples but not in the resume in Step 1.

        These are contents that are probably missing in the resume. 

        sample: {delimiter2}{samples}{delimiter2}

    Step 3: An resume advisor has also given some advices on what makes a bad resume and how to write good resume. The advises are delimited with {delimiter3} characters.
    
        Uses these advices to generate a list of suggestions for fields and content in previous steps. 

        For each badly written field or any missing fields, give your advice on how they can be improved and filled out. 

        advices: {delimiter3}{advices}{delimiter3}


    Use the following format:
        Step 1:{delimiter} <step 1 reasoning>
        Step 2:{delimiter} <step 2 reasoning>
        Step 3:{delimiter} <step 3 reasoning>


      Make sure to include {delimiter} to separate every step.
    
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    upgrade_resume_message = prompt_template.format_messages(
        resume = resume,
        samples = resume_samples,
        advices = resume_advices,
        job = my_job_title,
        job_description = job_description,
        delimiter = delimiter,
        delimiter1 = delimiter1, 
        delimiter2 = delimiter2,
        delimiter3 = delimiter3,
        delimiter4 = delimiter4, 
        
    )
    my_advice = llm(upgrade_resume_message).content

    # Check potential harmful content in response
    if (check_content_safety(text_str=my_advice)):   
        # Write the cover letter to a file
        with open(res_path, 'w') as f:
            try:
                f.write(my_advice)
                print("ALL SUCCESS")
            except Exception as e:
                print("FAILED")
                # Error logging



if __name__ == '__main__':
    basic_upgrade_resume(my_job_title)
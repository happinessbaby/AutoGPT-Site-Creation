import os
from openai_api import get_completion
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from basic_utils import read_txt
from common_utils import fetch_samples, get_web_resources
from samples import resume_samples_dict
from langchain.memory import ConversationBufferMemory
from langchain.agents import ConversationalChatAgent, Tool, AgentExecutor
from langchain_utils import create_QA_chain, create_qa_tools, create_custom_llm_agent
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

class ChatController(object):
    def __init__(self):
        self._create_chat_agent()

    def _create_chat_agent(self):

        self.llm = OpenAI(temperature=0, model_name="gpt-4", top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2)

        qa = create_QA_chain(self.llm, embeddings, "chroma" )

        tools = create_qa_tools(qa)

        # system_msg = "You are a helpful assistant."

        # prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
        # suffix = """Begin!"

        # {chat_history}
        # Question: {input}
        # {agent_scratchpad}"""

        # This probably can be changed to Custom Agent class
        # agent = ConversationalChatAgent.from_llm_and_tools(
        #     llm=self.llm,
        #     tools=tools,
        #     system_message=system_msg,
        #     prefix=prefix,
        #     suffix= suffix,
        # )
        agent = create_custom_llm_agent(self.llm, tools)

        self.chat_agent = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )

        # return self.chat_agent


    def askAI(self, id, question, callbacks=None):
        # try:
        #     objId = ObjectId(id)
        # except:
        #     raise HTTPException(status_code=400, detail="Not valid id.")

        # create a conversation memory and save it if it not exists 
        # can be changed to/incorporated into a streaming platform such as kafka
        if not os.path.isfile('conv_memory/'+id+'.pickle'):
            mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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
            response = self.chat_agent.run(question, callbacks=callbacks)
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




    

def basic_upgrade_resume(resume_file, my_job_title):

    resume = read_txt(resume_file)
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
    my_advices = llm(upgrade_resume_message).content
    print(my_advices)
    return my_advices




my_job_title = 'AI developer'
my_resume_file = 'resume_samples/resume2023v2.txt'

if __name__ == '__main__':
    basic_upgrade_resume(my_resume_file, my_job_title)
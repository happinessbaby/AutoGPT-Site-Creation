import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from pathlib import Path
import random
import time
import openai
import os
import uuid
from io import StringIO
from langchain.callbacks import StreamlitCallbackHandler
from career_advisor import ChatController
from callbacks.capturing_callback_handler import playback_callbacks
from basic_utils import convert_to_txt, read_txt, retrieve_web_content
from openai_api import check_content_safety
from dotenv import load_dotenv, find_dotenv
from common_utils import  check_content
import asyncio
import concurrent.futures
import subprocess
import sys
from multiprocessing import Process, Queue, Value
import pickle
import requests
from functools import lru_cache
from typing import Any
import multiprocessing as mp
from langchain.embeddings import OpenAIEmbeddings
from langchain_utils import retrieve_faiss_vectorstore, create_vectorstore, merge_faiss_vectorstore, create_retriever_tools




_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
upload_file_path = os.environ["UPLOAD_FILE_PATH"]
upload_link_path = os.environ["UPLOAD_LINK_PATH"]
placeholder = st.empty()
categories = ["resume", "cover letter", "job posting", "resume evaluation"]




class Chat(ChatController):



    def __init__(self):
        self._create_chatbot()

    # def _initialize_page(self):
    #     if 'page' not in st.session_state: st.session_state.page = 0
    # def nextPage(): st.session_state.page += 1
    # def firstPage(): st.session_state.page = 0

    def _create_chatbot(self):

        with placeholder.container():

            if "userid" not in st.session_state:
                st.session_state["userid"] = str(uuid.uuid4())
                print(f"Session: {st.session_state.userid}")
                # super().__init__(st.session_state.userid)
                new_chat = ChatController(st.session_state.userid)
                if "basechat" not in st.session_state:
                    st.session_state["basechat"] = new_chat

            try:
                self.new_chat = st.session_state.basechat
            except AttributeError as e:
                raise e

 
            # Initialize chat history
            # if "messages" not in st.session_state:
            #     st.session_state.messages = []



            # Set a default model
            # if "openai_model" not in st.session_state:
            #     st.session_state["openai_model"] = "gpt-3.5-turbo"

            # st.title("Career Advisor")

            # expand_new_thoughts = st.sidebar.checkbox(
            #     "Expand New Thoughts",
            #     value=True,
            #     help="True if LLM thoughts should be expanded by default",
            # )


            # Display chat messages from history on app rerun
            # for message in st.session_state.messages:
            #     with st.chat_message(message["role"]):
            #         st.markdown(message["content"])

        

            SAMPLE_QUESTIONS = {
                "":"",
                "help me write a cover letter": "coverletter",
                "help  evaluate my my resume": "resume",
                "I want to do a mock interview": "interview"
            }



            # Generate empty lists for generated and past.
            ## past stores User's questions
            if 'questions' not in st.session_state:
                st.session_state['questions'] = list()
            ## generated stores AI generated responses
            if 'responses' not in st.session_state:
                st.session_state['responses'] = list()

            question_container = st.container()
            results_container = st.container()


            # hack to clear text after user input
            if 'questionInput' not in st.session_state:
                st.session_state.questionInput = ''
            def submit():
                st.session_state.questionInput = st.session_state.input
                st.session_state.input = ''    
            # User input
            ## Function for taking user provided prompt as input
            def get_text():
                st.text_input("Ask a specific question: ", "", key="input", on_change = submit)
                return st.session_state.questionInput
            ## Applying the user input box
            with question_container:
                user_input = get_text()
                question_container = st.empty()
                st.session_state.questionInput=''


            #Sidebar section
            with st.sidebar:
                st.title('Career Help 🧸')
                # st.markdown('''
                # Hi, my name is Tebi, your AI career advisor. I can help you: 
                            
                # - improve your resume
                # - write a cover letter
                # - search for jobs
                                            
                # ''')

                add_vertical_space(3)

                # st.markdown('''
                #     Upload your resume and fill out a few questions for a quick start
                #             ''')
                with st.form( key='my_form', clear_on_submit=True):

                    # col1, col2, col3= st.columns([5, 1, 5])

                    # with col1:
                    #     job = st.text_input(
                    #         "job title",
                    #         "",
                    #         key="job",
                    #     )

                    #     company = st.text_input(
                    #         "company (optional)",
                    #         "",
                    #         key = "company"
                    #     )
                    
                    # with col2:
                    #     st.text('or')
                    
                    # with col3:

                    uploaded_files = st.file_uploader(label="Upload your file",
                                                       type=["pdf","odt", "docx","txt", "zip", "pptx", "ipynb"], 
                                                       accept_multiple_files=True)
                    add_vertical_space(3)
                    link = st.text_input("Share your link", "", key = "link")
                    add_vertical_space(1)
                    prefilled = st.selectbox(label="Quick navigation",
                                              options=sorted(SAMPLE_QUESTIONS.keys()), 
                                              label_visibility="hidden", 
                                              format_func=lambda x: '---General questions---' if x == '' else x)


                    submit_button = st.form_submit_button(label='Submit')

                    # if submit_button and uploaded_file is not None and (job is not None or posting is not None): 
                        # if "job" not in st.session_state:
                        #     st.session_state["job"] = job
                        # if "company" not in st.session_state:
                        #     st.session_state['company'] = company
                    if submit_button:
                        if uploaded_files:
                            self.process_file(uploaded_files)
                            # file_entities_list= self.process_file(uploaded_files)
                            # print(file_entities_list)
                            # if file_entities_list:
                            #     for file_entity in file_entities_list:
                            #         self.new_chat.update_entities(file_entity)
                  
                        if link:
                            self.process_link(link)
                            # link_q = Queue()
                            # link_p = Process(target = self.process_link, args=(link, link_q, ))
                            # link_p.start()
                            # link_p.join()
                            # link_entity = link_q.get()
                            # if link_entity != "":
                            #     self.new_chat.update_entities(link_entity)

                        if prefilled:
                            res = results_container.container()
                            streamlit_handler = StreamlitCallbackHandler(
                                parent_container=res,
                            )
                            question = prefilled
                            # answer = chat_agent.run(mrkl_input, callbacks=[streamlit_handler])
                            response = self.new_chat.askAI(st.session_state.userid, question, callbacks=streamlit_handler)
                            st.session_state.questions.append(question)
                            st.session_state.responses.append(response)


                add_vertical_space(3)

                # with st.form(key="question_selector"):
                #     prefilled = st.selectbox("Quick navigation", sorted(SAMPLE_QUESTIONS.keys())) or ""
                #     # question = st.text_input("Or, ask your own question", key=shadow_key)
                #     # st.session_state[key] = question
                #     # if not question:
                #         # question = prefilled
                #     submit_clicked = st.form_submit_button("Submit Question")

                #     if submit_clicked:
                #         res = results_container.container()
                #         streamlit_handler = StreamlitCallbackHandler(
                #             parent_container=res,
                #         )
                #         question = prefilled
                #         # answer = chat_agent.run(mrkl_input, callbacks=[streamlit_handler])
                #         response = self.new_chat.askAI(st.session_state.userid, question, callbacks=streamlit_handler)
                #         st.session_state.questions.append(question)
                #         st.session_state.responses.append(response)
                        # session_name = SAMPLE_QUESTIONS[question]



            # Chat section
            if user_input:
                res = results_container.container()
                streamlit_handler = StreamlitCallbackHandler(
                    parent_container=res,
                    # max_thought_containers=int(max_thought_containers),
                    # expand_new_thoughts=expand_new_thoughts,
                    # collapse_completed_thoughts=collapse_completed_thoughts,
                )
                question = user_input
                # answer = chat_agent.run(mrkl_input, callbacks=[streamlit_handler])
                response = self.new_chat.askAI(st.session_state.userid, question, callbacks = streamlit_handler)
                st.session_state.questions.append(question)
                st.session_state.responses.append(response)
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])-1, -1, -1):
                    message(st.session_state['responses'][i], key=str(i), avatar_style="initials", seed="AI")
                    message(st.session_state['questions'][i], is_user=True, key=str(i) + '_user',  avatar_style="initials", seed="Yueqi")



    def process_file(self, uploaded_files: Any) -> None:

        """ Processes user uploaded files including converting all format to txt, checking content safety, and categorizing content type  """

        for uploaded_file in uploaded_files:
            file_ext = Path(uploaded_file.name).suffix
            tmp_filename = str(uuid.uuid4())+file_ext
            tmp_save_path = os.path.join(upload_file_path, tmp_filename)
            with open(tmp_save_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            read_path =  os.path.join(upload_file_path, Path(tmp_filename).stem+'.txt')
            # Convert file to txt and save it to uploads
            convert_to_txt(tmp_save_path, read_path)
            content_safe, content_type, content_topics = check_content(read_path)
            print(content_type, content_safe, content_topics) 
            if content_safe:
                self.add_to_chat(content_type, content_topics, read_path)
            else:
                print("file content unsafe")
        

    def process_link(self, posting: Any) -> None:

        """ Processes user shared links including converting all format to txt, checking content safety, and categorizing content type """

        save_path = os.path.join(upload_link_path, str(uuid.uuid4())+".txt")
        if (retrieve_web_content(posting, save_path = save_path)):
            content_safe, content_type, content_topics = check_content(save_path)
            if content_safe:
                self.add_to_chat(content_type, content_topics, save_path)
            else:
                print("link content unsafe")


    def add_to_chat(self, content_type: str, content_topics: set, read_path: str) -> None:

        """ Updates entities, vector stores, and tools for different chat agents. The main chat agent will have file paths saved as entities. The interview agent will have files as tools.
        
        Args: 

            content_type (str): content category, such as resume, cover letter, job posting, other

            content_topics (str): if content_type is other, content will be treated as interview material. So in this case, content_topics is interview topics. 

            read_path (str): content file path
             
         """

        if content_type != "other":
            entity = f"""{content_type} file: {read_path}"""
            self.new_chat.update_entities(entity)
            name = content_type.strip().replace(" ", "_")
            vs_name = f"faiss_{name}_{st.session_state.userid}"
            vs = create_vectorstore("faiss", read_path, "file", vs_name)
            tool_name = vs_name.removeprefix("faiss_").removesuffix(f"_{self.userid}")
            tool_description =  """Useful for generating personalized interview questions and answers. 
                Use this tool more than any other tool during a mock interview session when there's a need to reference special content.
                Do not use this tool to load any files or documents.  """ 
            tools = create_retriever_tools(vs, tool_name, tool_description)
            self.new_chat.update_tools(tools, "interview")
        else:
            if content_topics:
                entity = f"""interview material topics: {str(content_topics)}"""
                self.new_chat.update_entities(entity)
            interview_material = f"faiss_interview_material_{st.session_state.userid}"
            vs = merge_faiss_vectorstore( interview_material, read_path)
            tool_name = vs_name.removeprefix("faiss_").removesuffix(f"_{self.userid}")
            tool_description =  """Useful for generating interview questions and answers. 
                Use this tool more than any other tool during a mock interview session. This tool can also be used for questions about topics in the interview material topics. 
                Do not use this tool to load any files or documents. This should be used only for searching and generating questions and answers of the relevant materials and topics. """
            tools = create_retriever_tools(vs, tool_name, tool_description)
            self.new_chat.update_tools(tools, "interview")



    



if __name__ == '__main__':

    advisor = Chat()
    # asyncio.run(advisor.initialize())
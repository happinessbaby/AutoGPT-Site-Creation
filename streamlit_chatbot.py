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
from mock_interview import InterviewController
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
from langchain_utils import retrieve_faiss_vectorstore, create_vectorstore, merge_faiss_vectorstore, create_vs_retriever_tools, create_retriever_tools
# import keyboard
from pynput.keyboard import Key, Controller
from pynput import keyboard
import sounddevice as sd
import soundfile as sf
import tempfile
import openai
from elevenlabs import generate, play, set_api_key
from time import gmtime, strftime
from playsound import playsound
from streamlit_modal import Modal
import json
from st_pages import show_pages_from_config, add_page_title, show_pages, Page

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar

# Optional -- adds the title and icon to the current page
# add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("streamlit_chatbot.py", "Home", "🏠"),
        Page("streamlit_interviewbot.py", "Mock Interview", ":books:"),
    ]
)





_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
set_api_key(os.environ["11LABS_API_KEY"])
save_path = os.environ["SAVE_PATH"]
temp_path = os.environ["TEMP_PATH"]
placeholder = st.empty()
duration = 5 # duration of each recording in seconds
fs = 44100 # sample rate
channels = 1 # number of channel
device = 4


class Chat():

    userid: str=""

    def __init__(self):
        self._create_chatbot()

  
    def _create_chatbot(self):

        with placeholder.container():

            if "userid" not in st.session_state:
                st.session_state["userid"] = str(uuid.uuid4())
                print(f"Session: {st.session_state.userid}")
                self.userid = st.session_state.userid
                # super().__init__(st.session_state.userid)
            if "basechat" not in st.session_state:
                new_chat = ChatController(st.session_state.userid)
                st.session_state["basechat"] = new_chat 
                
            try:
                self.new_chat = st.session_state.basechat
            except AttributeError as e:
                raise e
        


            # Initialize chat history
            # if "messages" not in st.session_state:
            #     st.session_state.messages = []


            # expand_new_thoughts = st.sidebar.checkbox(
            #     "Expand New Thoughts",
            #     value=True,
            #     help="True if LLM thoughts should be expanded by default",
            # )

        

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
                st.title('Quick Navigation 🧸')
                st.markdown('''
                Hi, I'm Acai, your AI career advisor. I can: 
                            
                - improve your resume
                - write a cover letter
                - conduct a mock interview 
                            
                Fill out the form below. Only your resume is really required, but I can generate better responses if you provide me with more information!
                                            
                ''')

                add_vertical_space(3)

                # st.markdown('''
                #     Upload your resume and fill out a few questions for a quick start
                #             ''')
                with st.form( key='my_form', clear_on_submit=True):

                    col1, col2= st.columns([5, 5])

                    with col1:
                        job = st.text_input(
                            "job title",
                            "",
                            key="job",
                        )
                    
                    with col2:
                        company = st.text_input(
                            "company name",
                            "",
                            key = "company"
                        )
                        
                    add_vertical_space(1)
                    link = st.text_input("job posting link", "", key = "link")


                    uploaded_files = st.file_uploader(label="Upload your file",
                                                    type=["pdf","odt", "docx","txt", "zip", "pptx"], 
                                                    key = "files",
                                                    accept_multiple_files=True)
                    add_vertical_space(1)
                    prefilled = st.selectbox(label="Quick navigation",
                                            options=sorted(SAMPLE_QUESTIONS.keys()), 
                                            label_visibility="hidden", 
                                            key = "prefilled",
                                            format_func=lambda x: '---Quick Help---' if x == '' else x)


                    submit_button = st.form_submit_button(label='Submit', on_click=self.form_callback)

                    # if submit_button and uploaded_file is not None and (job is not None or posting is not None): 
                    if submit_button:
                        # if job:
                        #     self.new_chat.update_entities(f"job:{job} \n ###")
                        # if company:
                        #     self.new_chat.update_entities(f"company:{company} \n ###")
                        # if uploaded_files:
                        #     self.process_file(uploaded_files)
                        # if link:
                        #     self.process_link(link)
                        if prefilled:
                            res = results_container.container()
                            streamlit_handler = StreamlitCallbackHandler(
                                parent_container=res,
                            )
                            question = prefilled
                            response = self.new_chat.askAI(st.session_state.userid, question, callbacks=streamlit_handler)
                            st.session_state.questions.append(question)
                            st.session_state.responses.append(response)


                add_vertical_space(3)



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
                response = self.new_chat.askAI(st.session_state.userid, question, callbacks = streamlit_handler)
                st.session_state.questions.append(question)
                st.session_state.responses.append(response)
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])-1, -1, -1):
                    message(st.session_state['responses'][i], key=str(i), avatar_style="initials", seed="AI", allow_html=True)
                    message(st.session_state['questions'][i], is_user=True, key=str(i) + '_user',  avatar_style="initials", seed="Yueqi", allow_html=True)


    # def mode_popup(self, mode):
    #     if mode=="interview":
    #         if "interview_state" not in st.session_state:
    #             modal = Modal(key="form_popup",title="""Your interview session will start. 
    #                           Please upload any interview material if you have not done so.""")
    #             with modal.container():
    #                 # with st.form( key='interview_form', clear_on_submit=True):
    #                 #     st.file_uploader(label="Upload your file",
    #                 #                                     type=["pdf","odt", "docx","txt", "zip", "pptx"], 
    #                 #                                     key = "interview_files",
    #                 #                                     accept_multiple_files=True)
    #                 #     st.form_submit_button(label='Submit', on_click=self.interview_form_callback)
    #                 #     # TODO: add links capability
    #                 col1, buffer, col2= st.columns([5, 5, 5])
    #                 with col1:
    #                     back = st.button("take me back to main chat")
    #                 with col2:
    #                     confirm = st.button("okay, take me to it!")
    #                 # with col3:
    #                 # skip = st.button("skip")
    #                 # if skip:
    #                 #     st.session_state.mode = "interview"
    #                 if back:
    #                     print("back is pressed")
    #                     st.session_state.mode = "chat"
    #                 if confirm:
    #                     print("confirm is pressed")          
    #                     st.session_state.mode = "interview"
    #                     st.experimental_rerun()
    #                 #TODO right now close button mode is still interview, need to switch to chat
    #                 # if modal.close():
    #                 #     st.session_state.mode = "chat"
    #     elif mode == "chat":
    #         if "chat_state" not in st.session_state:
    #             modal = Modal(key="popup_chat", title="Are you sure you want to go back to the main chat?")
    #             with modal.container():
    #                 col1, buffer, col2= st.columns([5, 5, 5])
    #                 with col1:
    #                     yes = st.button("yes")
    #                 with col2:
    #                     no = st.button("no")
    #                 if yes:
    #                     st.session_state.mode = "chat"
    #                     st.experimental_rerun()
    #                 if no:
    #                     st.session_state.mode = "interview"


    def form_callback(self):

        try: 
            temp_dir = temp_path+st.session_state.userid
            user_dir = save_path+st.session_state.userid
            os.mkdir(temp_dir)
            os.mkdir(user_dir)
        except FileExistsError:
            pass

        try:
            job = st.session_state.job
            self.new_chat.update_entities(f"job:{job} /n ###")
        except Exception:
            pass
        try: 
            company = st.session_state.company
            self.new_chat.update_entities(f"company:{company} /n ###")
        except Exception:
            pass
        try:
            files = st.session_state.files 
            self.process_file(files)
        except Exception:
            pass
        try:
            link = st.session_state.link
            self.process_link(link)
        except Exception:
            pass




    @st.cache()
    def process_file(self, uploaded_files: Any) -> None:

        """ Processes user uploaded files including converting all format to txt, checking content safety, and categorizing content type  """

        for uploaded_file in uploaded_files:
            file_ext = Path(uploaded_file.name).suffix
            filename = str(uuid.uuid4())+file_ext
            tmp_save_path = os.path.join(temp_path, st.session_state.userid, filename)
            with open(tmp_save_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            end_path =  os.path.join(save_path, st.session_state.userid, Path(filename).stem+'.txt')
            # Convert file to txt and save it to uploads
            convert_to_txt(tmp_save_path, end_path)
            content_safe, content_type, content_topics = check_content(end_path)
            print(content_type, content_safe, content_topics) 
            if content_safe:
                self.add_to_chat(content_type, content_topics, end_path)
            else:
                print("file content unsafe")
                os.remove(end_path)
        

    def process_link(self, posting: Any) -> None:

        """ Processes user shared links including converting all format to txt, checking content safety, and categorizing content type """

        end_path = os.path.join(save_path, st.session_state.userid, str(uuid.uuid4())+".txt")
        if (retrieve_web_content(posting, save_path = end_path)):
            content_safe, content_type, content_topics = check_content(end_path)
            if content_safe:
                self.add_to_chat(content_type, content_topics, end_path)
            else:
                print("link content unsafe")
                os.remove(end_path)


    def add_to_chat(self, content_type: str, content_topics: set, file_path: str) -> None:

        """ Updates entities, vector stores, and tools for different chat agents. The main chat agent will have file paths saved as entities. The interview agent will have files as tools.
        
        Args: 

            content_type (str): content category, such as resume, cover letter, job posting, other

            content_topics (str): if content_type is other, content will be treated as interview material. So in this case, content_topics is interview topics. 

            read_path (str): content file path
             
         """

        if content_type != "other":
            entity = f"""{content_type} file: {file_path} /n ###"""
            self.new_chat.update_entities(entity)
            # name = content_type.strip().replace(" ", "_")
            # vs_name = f"faiss_{name}_{st.session_state.userid}"
            # vs = create_vectorstore("faiss", file_path, "file", vs_name)
            # tool_name = "search_" + vs_name.removeprefix("faiss_").removesuffix(f"_{st.session_state.userid}")
            # tool_description =  f"""Useful for searching documents with respect to content type {content_type}.
            #     Do not use this tool to load any files or documents.  """ 
            # tools = create_retriever_tools(vs.as_retriever(), tool_name, tool_description)
            # self.new_chat.update_tools(tool_name)
   
        else:
            if content_topics: 
                entity = f"""study material topics: {str(content_topics)} """
                self.new_chat.update_entities(entity)
            # vs_name = f"faiss_study_material_{st.session_state.userid}"
            # vs = merge_faiss_vectorstore(vs_name, file_path)
            # tool_name = "search_" + vs_name.removeprefix("faiss_").removesuffix(f"_{st.session_state.userid}")
            # tool_description =  f"""Used for searching material with respect to content topics such as {content_topics}. 
            #     Do not use this tool to load any files or documents. This should be used only for searching and generating questions and answers of the relevant materials and topics. """
            # tools = create_retriever_tools(vs.as_retriever(), tool_name, tool_description)
            # self.new_chat.update_tools(tools)
            

    



if __name__ == '__main__':

    advisor = Chat()
    # asyncio.run(advisor.initialize())

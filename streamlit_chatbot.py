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
from basic_utils import convert_to_txt, read_txt, retrieve_web_content, html_to_text
from openai_api import get_completion, num_tokens_from_text
from openai_api import check_content_safety
from dotenv import load_dotenv, find_dotenv
from common_utils import  check_content, evaluate_content, generate_tip_of_the_day, shorten_content
import asyncio
import concurrent.futures
import subprocess
import sys
import re
from multiprocessing import Process, Queue, Value
import pickle
import requests
from functools import lru_cache
from typing import Any
import multiprocessing as mp
from langchain_utils import retrieve_faiss_vectorstore, create_vectorstore, merge_faiss_vectorstore, create_vs_retriever_tools, create_retriever_tools, create_tag_chain
import openai
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
        # Page("streamlit_resources.py", "Resources", ":busts_in_silhouette:" ),
    ]
)





_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
save_path = os.environ["SAVE_PATH"]
temp_path = os.environ["TEMP_PATH"]
placeholder = st.empty()
duration = 5 # duration of each recording in seconds
fs = 44100 # sample rate
channels = 1 # number of channel
device = 4
max_token_count=3500
topic = "tech jobs"


class Chat():

    userid: str=""

    def __init__(self):
        self._create_chatbot()

  
    def _create_chatbot(self):

        styl = f"""
        <style>
            .stTextInput {{
            position: fixed;
            bottom: 3rem;
            }}
        </style>
        """
        # styl = f"""
        # <style>
        #     .element-container:nth-of-type(1) stTextInput {{
        #     position: fixed;
        #     bottom: 3rem;
        #     }}
        # </style>
        # """
        st.markdown(styl, unsafe_allow_html=True)

        with placeholder.container():

            if "userid" not in st.session_state:
                st.session_state["userid"] = str(uuid.uuid4())
                print(f"Session: {st.session_state.userid}")
                self.userid = st.session_state.userid
                # super().__init__(st.session_state.userid)
            if "basechat" not in st.session_state:
                new_chat = ChatController(st.session_state.userid)
                st.session_state["basechat"] = new_chat 
            if "tipofday" not in st.session_state:
                tip = generate_tip_of_the_day(topic)
                st.session_state["tipofday"] = tip
                st.write(tip)
                
            try:
                self.new_chat = st.session_state.basechat
            except AttributeError as e:
                raise e
        
            try: 
                temp_dir = temp_path+st.session_state.userid
                user_dir = save_path+st.session_state.userid
                os.mkdir(temp_dir)
                os.mkdir(user_dir)
            except FileExistsError:
                pass

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
                "help me write a cover letter": "generate",
                "help me evaluate my resume": "evaluate",
                "help me reformat my resume": "reformat",
                "help me rewrite my document": "customize"
            }



            # Generate empty lists for generated and past.
            ## past stores User's questions
            if 'questions' not in st.session_state:
                st.session_state['questions'] = list()
            ## generated stores AI generated responses
            if 'responses' not in st.session_state:
                st.session_state['responses'] = list()
            # if "responses_container" not in st.session_state:
            #     st.session_state["responses_container"] = st.container()
            # if "questions_container" not in st.session_state:
            #     st.session_state["questions_container"] = st.container()
            # question_container = st.container()
            # results_container = st.container()


            # hack to clear text after user input
            if 'questionInput' not in st.session_state:
                st.session_state.questionInput = ''
            # def submit():
            #     st.session_state.questionInput = st.session_state.input
            #     st.session_state.input = ''    
            # User input
            ## Function for taking user provided prompt as input
            # def get_text():
            #     st.text_input("Chat with me: ", "", key="input", on_change = submit)
            #     return st.session_state.questionInput
            ## Applying the user input box
            # with st.session_state.questions_container:
            #     st.text_input("Chat with me: ", "", key="input", on_change = submit)
                # user_input = get_text()
                # st.session_state.questions_container = st.empty()
                # st.session_state.questionInput=''


            #Sidebar section
            with st.sidebar:
                st.title("""Hi, I'm Acai 🧸""")
                st.markdown('''
                I'm a career AI advisor. I can:
                            
                - Help with your resume, cover letter, and personal statement
                - Conduct mock interview

                If you want to practice for your next interview, please click on the Mock Interview tab above. 
                Otherwise, start with the Quick Navigation below, or just chat with me!
                                            
                ''')

                add_vertical_space(3)

                st.markdown("Quick Navigation")
                with st.form( key='my_form', clear_on_submit=True):

                    # col1, col2= st.columns([5, 5])

                    # with col1:
                    #     job = st.text_input(
                    #         "job/program",
                    #         "",
                    #         key="job",
                    #     )
                    
                    # with col2:
                    #     company = st.text_input(
                    #         "company/institution",
                    #         "",
                    #         key = "company"
                    #     )

                    # about_me = st.text_area(label="About", placeholder="""You can say,  I want to apply for the MBA program at ABC University, or I wish to work for XYZ as a customer service representative. 
                    
                    # If you want to provide any links, such as a link to a job posting, please paste it here. """, key="about_me")

                    uploaded_files = st.file_uploader(label="Upload your files",
                                                      help = "This can be your resume, cover letter, or anything else you want to provide me with. ",
                                                    type=["pdf","odt", "docx","txt", "zip", "pptx"], 
                                                    key = "files",
                                                    accept_multiple_files=True)

                    
                    link = st.text_area(label="Paste your link", key = "link", help="This can be a job posting, for example.")

                    add_vertical_space(1)
                    prefilled = st.selectbox(label="Ask me a question",
                                            options=sorted(SAMPLE_QUESTIONS.keys()), 
                                            key = "prefilled",
                                            format_func=lambda x: '' if x == '' else x,
                                            )


                    submit_button = st.form_submit_button(label='Submit', on_click=self.form_callback)

                    # if submit_button:
                    #     # if job:
                    #     #     self.new_chat.update_entities(f"job:{job} \n ###")
                    #     # if company:
                    #     #     self.new_chat.update_entities(f"company:{company} \n ###")
                    #     # if uploaded_files:
                    #     #     self.process_file(uploaded_files)
                    #     # if link:
                    #     #     self.process_link(link)
                    #     if prefilled:
                    #         # res = st.session_state.responses_container.container()
                    #         # streamlit_handler = StreamlitCallbackHandler(
                    #         #     parent_container=res,
                    #         # )
                    #         self.question = prefilled
                    #         response = self.new_chat.askAI(st.session_state.userid, self.question, callbacks=None)
                    #         st.session_state.questions.append(self.question)
                    #         st.session_state.responses.append(response)

                # upload_error = st.empty()
                # if "upload_error" not in st.session_state:
                #     st.session_state["upload_error"] = upload_error

                add_vertical_space(3)

            # Chat section
            # if user_input:
            # if st.session_state.questionInput:
            #     # res = st.session_state.responses_container.container()
            #     # streamlit_handler = StreamlitCallbackHandler(
            #     #     parent_container=res,
            #     #     # max_thought_containers=int(max_thought_containers),
            #     #     # expand_new_thoughts=expand_new_thoughts,
            #     #     # collapse_completed_thoughts=collapse_completed_thoughts,

            #     # self.question = user_input
            #     user_input = st.session_state.questionInput
            #     self.question = self.process_user_input(user_input)
            #     response = self.new_chat.askAI(st.session_state.userid, self.question, callbacks = None)
            #     st.session_state.questions.append(self.question)
            #     st.session_state.responses.append(response)
            #     st.session_state.questionInput=''

            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])-1, -1, -1):
                    message(st.session_state['responses'][i], key=str(i), avatar_style="initials", seed="ACAI", allow_html=True)
                    message(st.session_state['questions'][i], is_user=True, key=str(i) + '_user',  avatar_style="initials", seed="Yueqi", allow_html=True)

            st.text_input("Chat with me: ", "", key="input", on_change = self.chatbox_callback)

    def chatbox_callback(self):

        """ Sends user input from chatbox and prefilled question selection to chat agent. """
          
        self.question = self.process_user_input(st.session_state.input)
        if self.question:
            response = self.new_chat.askAI(st.session_state.userid, self.question, callbacks = None)
            st.session_state.questions.append(st.session_state.input)
            st.session_state.responses.append(response)
        st.session_state.questionInput = st.session_state.input
        st.session_state.input = ''    




    def form_callback(self):

        """ Processes form information after form submission. """

        # try:
        #     job = st.session_state.job
        #     self.new_chat.update_entities(f"job:{job} /n ###")
        # except Exception:
        #     pass
        # try: 
        #     company = st.session_state.company
        #     self.new_chat.update_entities(f"company:{company} /n ###")
        # except Exception:
        #     pass
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
        # try:
        #     about_me = st.session_state.about_me
        #     self.process_about_me(about_me)
        # except Exception:
        #     pass
        if st.session_state.prefilled:
            st.session_state.input = st.session_state.prefilled
            self.chatbox_callback()
        else:
        # passes the previous user question to the agent one more time after user uploads form
            try:
                print(f"QUESTION INPUT: {st.session_state.questionInput}")
                # res = st.session_state.responses_container.container()
                # streamlit_handler = StreamlitCallbackHandler(
                #     parent_container=res,
                # )
                if st.session_state.questionInput!="":
                    response = self.new_chat.askAI(st.session_state.userid, st.session_state.questionInput, callbacks=None)
                    st.session_state.questions.append(st.session_state.questionInput)
                    st.session_state.responses.append(response)
            # 'Chat' object has no attribute 'question': exception occurs when user has not asked a question, in this case, pass
            except AttributeError:
                pass
                
    def process_user_input(self, user_input: str) -> str:

        """ Processes user input and processes any links in the input. """
        tag_schema = {
            "properties": {
                "aggressiveness": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                    "description": "describes how aggressive the statement is, the higher the number the more aggressive",
                },
                "topic": {
                    "type": "string",
                    "enum": ["self description", "job or program description", "company or institution description",],
                    "description": "determines if the statement contains certain topic",
                },
            },
            "required": ["topic", "sentiment", "aggressiveness"],
        }
        response = create_tag_chain(tag_schema, user_input)
        topic = response.get("topic", "")
        if topic != "other":
            self.new_chat.update_entities(f"about me:{user_input} /n ###")
        urls = re.findall(r'(https?://\S+)', user_input)
        print(urls)
        if urls:
            for url in urls:
                self.process_link(url)
        return user_input
        
        if check_content_safety(text_str=user_input):
            if evaluate_content(user_input, "a job, program, company, or institutation description or a personal background description"):
                self.new_chat.update_entities(f"about me:{user_input} /n ###")
            urls = re.findall(r'(https?://\S+)', user_input)
            print(urls)
            if urls:
                for url in urls:
                    self.process_link(url)
            return user_input
        else: return ""




    # def process_about_me(self, about_me: str) -> None:
    
    #     """ Processes user's about me input for content type and processes any links in the description. """

    #     content_type = """a job or study related user request. """
    #     user_request = evaluate_content(about_me, content_type)
    #     about_me_summary = get_completion(f"""Summarize the following about me, if provided, and ignore all the links: {about_me}. """)
    #     self.new_chat.update_entities(f"about me:{about_me_summary} /n ###")
    #     if user_request:
    #         self.question = about_me
    #     # process any links in the about me
    #     urls = re.findall(r'(https?://\S+)', about_me)
    #     print(urls)
    #     if urls:
    #         for url in urls:
    #             self.process_link(url)




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
            if content_safe and content_type!="empty":
                self.update_entities(content_type, content_topics, end_path)
            else:
                st.error(f"Failed processing {Path(uploaded_file.name).root}. Please try another file!")
                # st.session_state.upload_error.markdown(f"{Path(uploaded_file.name).root} failed. Please try another file!")
                os.remove(end_path)
        

    def process_link(self, link: Any) -> None:

        """ Processes user shared links including converting all format to txt, checking content safety, and categorizing content type """

        end_path = os.path.join(save_path, st.session_state.userid, str(uuid.uuid4())+".txt")
        if html_to_text([link], save_path=end_path):
        # if (retrieve_web_content(posting, save_path = end_path)):
            content_safe, content_type, content_topics = check_content(end_path)
            print(content_type, content_safe, content_topics) 
            if content_safe and content_type!="empty" and content_type!="browser error":
                self.update_entities(content_type, content_topics, end_path)
            else:
                st.error(f"Failed processing {link}. Please try another link!")
                # st.session_state.upload_error.markdown(f"{link} failed. Please try another link!")
                os.remove(end_path)


    def update_entities(self, content_type:str, content_topics:str, end_path:str) -> str:

        """ Update entities for chat agent. """

        # if content_type=="browser error" or content_type=="empty":
        #     modal = Modal(key="error_popup", title="Content cannot be processed")
        #     with modal.container():
        #         st.write("If you shared a link, please try pasting the link content, or save the content as file and try again.")
        if content_type!="other" and content_type!="learning material":
            if content_type=="job posting":
                content = read_txt(end_path)
                token_count = num_tokens_from_text(content)
                if token_count>max_token_count:
                    shorten_content(end_path, content_type) 
            content_type = content_type.replace(" ", "_").strip()        
            entity = f"""{content_type}_file: {end_path} /n ###"""
            self.new_chat.update_entities(entity)
        if content_type=="learning material" :
            # update user material, to be used for "search_user_material" tool
            self.update_vectorstore(content_topics, end_path)


    def update_vectorstore(self, content_topics: str, end_path: str) -> None:

        """ Update vector store for chat agent. """

        entity = f"""topics: {str(content_topics)} """
        self.new_chat.update_entities(entity)
        vs_name = "user_material"
        vs = merge_faiss_vectorstore(vs_name, end_path)
        vs_path = f"./faiss/{st.session_state.userid}/chat/{vs_name}"
        vs.save_local(vs_path)
        entity = f"""user_material_path: {vs_path} /n ###"""
        self.new_chat.update_entities(entity)



    # def add_to_chat(self, content_type: str, content_topics: set, file_path: str) -> None:

    #     """ Updates entities, vector stores, and tools for different chat agents. The main chat agent will have file paths saved as entities. The interview agent will have files as tools.
        
    #     Args: 

    #         content_type (str): content category, such as resume, cover letter, job posting, other

    #         content_topics (str): if content_type is other, content will be treated as interview material. So in this case, content_topics is interview topics. 

    #         read_path (str): content file path
             
    #      """

    #     if content_type != "other":
    #         entity = f"""{content_type} file: {file_path} /n ###"""
    #         self.new_chat.update_entities(entity)
    #         # name = content_type.strip().replace(" ", "_")
    #         # vs_name = f"faiss_{name}_{st.session_state.userid}"
    #         # vs = create_vectorstore("faiss", file_path, "file", vs_name)
    #         # tool_name = "search_" + vs_name.removeprefix("faiss_").removesuffix(f"_{st.session_state.userid}")
    #         # tool_description =  f"""Useful for searching documents with respect to content type {content_type}.
    #         #     Do not use this tool to load any files or documents.  """ 
    #         # tools = create_retriever_tools(vs.as_retriever(), tool_name, tool_description)
    #         # self.new_chat.update_tools(tool_name)
   
    #     else:
    #         if content_topics: 
    #             entity = f"""study material topics: {str(content_topics)} """
    #             self.new_chat.update_entities(entity)
    #         # vs_name = f"faiss_study_material_{st.session_state.userid}"
    #         # vs = merge_faiss_vectorstore(vs_name, file_path)
    #         # tool_name = "search_" + vs_name.removeprefix("faiss_").removesuffix(f"_{st.session_state.userid}")
    #         # tool_description =  f"""Used for searching material with respect to content topics such as {content_topics}. 
    #         #     Do not use this tool to load any files or documents. This should be used only for searching and generating questions and answers of the relevant materials and topics. """
    #         # tools = create_retriever_tools(vs.as_retriever(), tool_name, tool_description)
    #         # self.new_chat.update_tools(tools)
            

    



if __name__ == '__main__':

    advisor = Chat()
    # asyncio.run(advisor.initialize())

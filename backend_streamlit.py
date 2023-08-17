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
from langchain_utils import split_doc,merge_faiss_vectorstore, retrieve_faiss_vectorstore, create_vectorstore
from common_utils import categorize_content
import asyncio
import concurrent.futures
import subprocess
import sys
from multiprocessing import Process, Queue, Value
import pickle
import requests
from functools import lru_cache
import multiprocessing as mp
from langchain.embeddings import OpenAIEmbeddings




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
                # "What are some general advices for writing an outstanding resume?": "general_advices.pickle",
                # "What are some things I could be doing terribly wrong with my resume?": "bad_resume.pickle",
                "help me write a cover letter": "coverletter",
                "help  evaluate my my resume": "resume",
                "help me do a mock interview": "interview"
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


            #Sidebar section
            with st.sidebar:
                st.title('Career Chat 🧸')
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
                    link = st.text_input(
                        "input your link",
                        "",
                        key = "link"
                    )

                    uploaded_file = st.file_uploader(label="Upload your file", type=["pdf","odt", "docx","txt"])
                    # if uploaded_file is not None:
                    #     # To convert to a string based IO:
                    #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        # st.write(stringio)

                    submit_button = st.form_submit_button(label='Submit')

                    # if submit_button and uploaded_file is not None and (job is not None or posting is not None): 
                        # if "job" not in st.session_state:
                        #     st.session_state["job"] = job
                        # if "company" not in st.session_state:
                        #     st.session_state['company'] = company
                    if submit_button:
                        if uploaded_file:
                            file_q = Queue()
                            file_p = Process(target=self.process_file, args=(uploaded_file, file_q, ))
                            file_p.start()
                            file_p.join()
                            file_entity = file_q.get()
                            if file_entity !="":
                                self.new_chat.update_entities(file_entity)
                            
                        if link:
                            link_q = Queue()
                            link_p = Process(target = self.process_link, args=(link, link_q, ))
                            link_p.start()
                            link_p.join()
                            link_entity = link_q.get()
                            if link_entity != "":
                                self.new_chat.update_entities(link_entity)

                add_vertical_space(3)

                with st.form(key="question_selector"):
                    prefilled = st.selectbox("Quick navigation", sorted(SAMPLE_QUESTIONS.keys())) or ""
                    # question = st.text_input("Or, ask your own question", key=shadow_key)
                    # st.session_state[key] = question
                    # if not question:
                        # question = prefilled
                    submit_clicked = st.form_submit_button("Submit Question")

                    if submit_clicked:
                        res = results_container.container()
                        streamlit_handler = StreamlitCallbackHandler(
                            parent_container=res,
                        )
                        question = prefilled
                        # answer = chat_agent.run(mrkl_input, callbacks=[streamlit_handler])
                        response = self.new_chat.askAI(st.session_state.userid, question, callbacks=streamlit_handler)
                        st.session_state.questions.append(question)
                        st.session_state.responses.append(response)
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
                user_input = ""
            if st.session_state['responses']:
                # user_input = ""
                for i in range(len(st.session_state['responses'])):
                    message(st.session_state['questions'][i], is_user=True, key=str(i) + '_user',  avatar_style="initials", seed="Yueqi")
                    message(st.session_state['responses'][i], key=str(i), avatar_style="initials", seed="AI")


    def process_file(self, uploaded_file, queue):
        file_ext = Path(uploaded_file.name).suffix
        tmp_filename = str(uuid.uuid4())+file_ext
        tmp_save_path = os.path.join(upload_file_path, tmp_filename)
        with open(tmp_save_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        read_path =  os.path.join(upload_file_path, Path(tmp_filename).stem+'.txt')
        # Convert file to txt and save it to uploads
        convert_to_txt(tmp_save_path, read_path)
        if (check_content_safety(file=read_path)):
            content_type = categorize_content(read_txt(read_path))
            if (content_type in categories):
                print(f"file content type: {content_type}")
                    #TODO: for logged in users, user_path will not be session state variable anymore,
                    #  and all their original uploaded files and links should be saved 
                if content_type=="resume":
                    new_entity = f"""resume file: {read_path}"""
                elif content_type == "cover letter":
                    new_entity = f"""cover letter file: {read_path}"""
                elif content_type == "resume evaluation":
                    new_entity = f"""resume evaluation file: {read_path}"""
                queue.put(new_entity)
            else:
                print("file content rejected")
                queue.put("")
        else:
            print("file content unsafe")
            queue.put("")

    def process_link(self, posting,  queue):
        tmp_save_path = os.path.join(upload_link_path, str(uuid.uuid4())+".txt")
        if (retrieve_web_content(posting, save_path = tmp_save_path)):
            content_type = categorize_content(read_txt(tmp_save_path))
            if (content_type in categories):
                print(f"link content type: {content_type}")
                if content_type == "job posting":
                    new_entity = f"""job posting link: {tmp_save_path}"""
                    queue.put(new_entity)
            else:
                print("link content rejected")
                queue.put("")
        else:
            print("link content unsafe")
            queue.put("")
    






    



    # def show_progress(self):

    #     with placeholder.container():

    #         progress_text = "Your career advisor is on its way."
    #         my_bar = st.progress(0, text=progress_text)
    #         # to do replaced with real background progress
    #         for percent_complete in range(100):
    #             time.sleep(1)
    #             my_bar.progress(percent_complete + 1, text=progress_text)
    #             if Path(os.path.join(advice_path, self.id+".txt")).is_file():
    #                 self.create_chatbot()

            # st.spinner('Your assistant is on its way...')
            # timer=60
            # while timer>0:
            #     if Path(self.advice_file).is_file():
            #         st.success('Done!')
            #         self.create_chatbot()
            #     timer-=1


    



if __name__ == '__main__':
    # create_chatbot()
    # with mp.Manager() as manager:
    #     chat_agents = manager.dict()
    #     p = mp.Process(target=initialize_chats, args=(chat_agents,))
    #     p.start()
    #     p.join()
    advisor = Chat()
    # asyncio.run(advisor.initialize())
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
from basic_utils import convert_to_txt, check_content_safety, read_txt
from upgrade_resume import evaluate_resume
from dotenv import load_dotenv, find_dotenv
import asyncio
import concurrent.futures
import subprocess
import sys
from multiprocessing import Process, Queue, Value
import pickle
import requests
from generate_cover_letter import generate_basic_cover_letter


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

placeholder = st.empty()
upload_path = "./uploads"
cover_letter_path = "./static/cover_letter"
advice_path = "./static/advice"


class Chat():

    def __init__(self):
        pass
        # self.id = "advice"
        # self.resume = None
        # self.job = None
        # self.company = None


    # def _initialize_page(self):
    #     if 'page' not in st.session_state: st.session_state.page = 0
    # def nextPage(): st.session_state.page += 1
    # def firstPage(): st.session_state.page = 0

    def create_chatbot(self):

        with placeholder.container():

            # st_callback = StreamlitCallbackHandler(st.container())
            if "userid" not in st.session_state:
                st.session_state["userid"] = str(uuid.uuid4())
                print(st.session_state.userid)

            new_chat = ChatController()


            # chat_agent = new_chat.create_chat_agent()

            # Initialize chat history
            # if "messages" not in st.session_state:
            #     st.session_state.messages = []


            # Check if 'key' already exists in session_state
            # If not, then initialize it
            # if 'key' not in st.session_state:
            #     st.session_state['key'] = 'value'

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

            # Accept user input
            # if prompt := st.chat_input("What is up?"):
                # # Add user message to chat history
                # st.session_state.messages.append({"role": "user", "content": prompt})
                # # Display user message in chat message container
                # with st.chat_message("user"):
                #     st.markdown(prompt)
                # # Display assistant response in chat message container
                # with st.chat_message("assistant"):
                #     message_placeholder = st.empty()
                #     full_response = ""
                #     for response in openai.ChatCompletion.create(
                #         model=st.session_state["openai_model"],
                #         messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                #             stream=True,
                #      ):
                #         full_response += response.choices[0].delta.get("content", "")
                #         message_placeholder.markdown(full_response + "▌")
                #     message_placeholder.markdown(full_response)
                # st.session_state.messages.append({"role": "assistant", "content": full_response})

            SAMPLE_QUESTIONS = {
                # "What are some general advices for writing an outstanding resume?": "general_advices.pickle",
                # "What are some things I could be doing terribly wrong with my resume?": "bad_resume.pickle",
                "help me write a cover letter": "coverletter"
            }

            # key = "input"
            # shadow_key = "_input"
            # key = self.id
            # shadow_key = f"_{self.id}"


            # if key in st.session_state and shadow_key not in st.session_state:
            #     st.session_state[shadow_key] = st.session_state[key]



            # Generate empty lists for generated and past.
            ## past stores User's questions
            if 'questions' not in st.session_state:
                st.session_state['questions'] = [""]
            ## generated stores AI generated responses
            if 'responses' not in st.session_state:
                st.session_state['responses'] = [f"Hi, I'm your career advisor Tebi. How can I help you?"]

            # question_container = st.empty()
            # results_container = st.empty()
            question_container = st.container()
            results_container = st.container()


            # User input
            ## Function for taking user provided prompt as input
            def get_text():
                input_text = st.text_input("Ask a specific question: ", "", key="input")
                return input_text
            ## Applying the user input box
            with question_container:
                user_input = get_text()

            #Sidebar section
            with st.sidebar:
                st.title('Career Chat 🧸')
                st.markdown('''
                Hi, my name is Tebi, your AI career advisor. I can help you: 
                            
                - improve your resume
                - write a cover letter
                - search for jobs
                                            
                ''')

                add_vertical_space(3)

                st.markdown('''
                    Upload your resume and fill out a few questions for a quick start
                            ''')
                       # A hack to "clear" the previous result when submitting a new prompt.
                with st.form(key='my_form', clear_on_submit=True):

                    job = st.text_input(
                        "what position are you applying for?",
                        "",
                        key="job",
                    )

                    company = st.text_input(
                        "what company/companies do you want to work for?",
                        "",
                        key = "company"
                    )

                    uploaded_file = st.file_uploader(label="Upload your resume", type=["pdf","odt", "docx","txt"])
                    # if uploaded_file is not None:
                    #     # To convert to a string based IO:
                    #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        # st.write(stringio)

                    submit_button = st.form_submit_button(label='Submit')
                    from clear_results import with_clear_container
                    if submit_button and uploaded_file is not None and job is not None:
                        if "job" not in st.session_state:
                            st.session_state["job"] = job
                        if "company" not in st.session_state:
                            st.session_state['company'] = company
                        # loop = asyncio.get_running_loop()
                        # 3. Run in a custom process pool:
                        # with concurrent.futures.ProcessPoolExecutor() as pool:
                        #     result = await loop.run_in_executor(pool, self.assess(uploaded_file, text_input))
                        #     print('custom process pool', result)
                        # queue = Queue()
                        # self.id = str(uuid.uuid4())
                        # userid = self.id
                        # self.job = job
                        # self.company = company
                        p = Process(target=self.assess, args=(st.session_state.userid, uploaded_file, st.session_state.job,))
                        p.start()
                        # subprocess.run([f"{sys.executable}", "assess.py"])
                        # self.show_progress()
                        p.join() # this blocks until the process terminates
                        # result = queue.get()
                        # print(result)

                add_vertical_space(5)

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
                        session_name = SAMPLE_QUESTIONS[question]
                        if session_name == "coverletter":
                            read_path = os.path.join(upload_path, st.session_state.userid+'.txt')
                            if Path(read_path).is_file():
                                save_path = os.path.join(cover_letter_path, st.session_state.userid+'.txt')
                                p = Process(target=generate_basic_cover_letter, args=(st.session_state.job, st.session_state.company, read_path, save_path))
                                p.start()
                                # progress feedbacks
                                p.join()
                                response = read_txt(save_path)
                            else:
                                print(st.session_state.job)
                                print(st.session_state.userid)
                                response = "Sure! Just uploade your resume and I'll give help you with it. "
                        else: 
                            session_path = Path(__file__).parent / "general_questions" / session_name
                            print(f"Playing saved session: {session_path}")
                            response = playback_callbacks(
                                [streamlit_handler], str(session_path), max_pause_time=3
                            )         
                        st.session_state.questions.append(question)
                        st.session_state.responses.append(response)


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
                response = new_chat.askAI(st.session_state.userid, question, callbacks=[streamlit_handler])
                st.session_state.questions.append(question)
                st.session_state.responses.append(response)

            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    message(st.session_state['questions'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state['responses'][i], key=str(i))

                




    # def initialize(self):

    #     with placeholder.container():

    #         with st.form(key='my_form'):

    #             text_input = st.text_input(
    #                 "Tell me about your dream job",
    #                 "",
    #                 key="job",
    #             )

    #             uploaded_file = st.file_uploader(label="Upload your resume", type=["pdf","odt", "docx","txt"])
    #             # if uploaded_file is not None:
    #             #     # To convert to a string based IO:
    #             #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #                 # st.write(stringio)

    #             submit_button = st.form_submit_button(label='Send to resume advisor')
    #             if submit_button:
    #                 self.id = str(uuid.uuid4())
    #                 p = Process(target=self.assess, args=(uploaded_file, text_input,))
    #                 p.start()
    #                 # subprocess.run([f"{sys.executable}", "assess.py"])
    #                 self.show_progress()
    #                 p.join() # this blocks until the process terminates
    #                 # result = queue.get()
    #                 # print(result)
                        

    def assess(self, userid, uploaded_file, text_input,):
        file_ext = Path(uploaded_file.name).suffix
        filename = userid+file_ext
        # uploaded_file.name = filename
        save_path = os.path.join(upload_path, filename)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        read_path =  os.path.join(upload_path, Path(filename).stem+'.txt')
        convert_to_txt(save_path, read_path)
        if (Path(read_path).exists()):
            # Check for content safety
            if (check_content_safety(file=read_path)):
                res_path = os.path.join(advice_path, os.path.basename(read_path))
                evaluate_resume(text_input, read_path = read_path, res_path = res_path)



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
    advisor = Chat()
    # asyncio.run(advisor.initialize())
    # advisor.initialize()
    advisor.create_chatbot()
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


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

placeholder = st.empty()
upload_path = "./uploads"
result_path = "./static"




class Chat():

    def __init__(self):
        self.id = "advice"


    # def _initialize_page(self):
    #     if 'page' not in st.session_state: st.session_state.page = 0
    # def nextPage(): st.session_state.page += 1
    # def firstPage(): st.session_state.page = 0


    def create_chatbot(self):

        with placeholder.container():

            # st_callback = StreamlitCallbackHandler(st.container())
            file = os.path.join(result_path, self.id+".txt")
            new_chat = ChatController(file)
            # chat_agent = new_chat.create_chat_agent()

            # Initialize chat history
            # if "messages" not in st.session_state:
            #     st.session_state.messages = []


            # Check if 'key' already exists in session_state
            # If not, then initialize it
            if 'key' not in st.session_state:
                st.session_state['key'] = 'value'

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
                "What are some general advices for writing an outstanding resume?": "./general_questions/general_advices.pickle",
                "What are some things I could be doing terribly wrong with my resume?": "./general_questions/bad_resume.pickle",
                "help me write a cover letter?": "./general_questions/coverletter.pickle"
            }

            # key = "input"
            # shadow_key = "_input"
            key = self.id
            shadow_key = f"_{self.id}"


            if key in st.session_state and shadow_key not in st.session_state:
                st.session_state[shadow_key] = st.session_state[key]


            with st.sidebar:
                st.title('Career Chat 🧸')
                st.markdown('''
                ## About
                Hi, my name is Tebi, your AI career advisor. I can help you: 
                            
                - Upgrade your resume
                - Write a cover letter
                - Search for a open job position
                
                Pick from the list of general questions for quick navigation. ''')
                add_vertical_space(5)
                # st.write('Made with ❤️ (<https://youtube.com/dataprofessor>)')

                with st.form(key="question_selector"):
                    prefilled = st.selectbox("General questions", sorted(SAMPLE_QUESTIONS.keys())) or ""
                    # question = st.text_input("Or, ask your own question", key=shadow_key)
                    # st.session_state[key] = question
                    # if not question:
                        # question = prefilled
                    submit_clicked = st.form_submit_button("Submit Question")

            # Generate empty lists for generated and past.
            ## past stores User's questions
            if 'questions' not in st.session_state:
                st.session_state['questions'] = [""]
            ## generated stores AI generated responses
            if 'responses' not in st.session_state:
                advice = read_txt(file)
                st.session_state['responses'] = [f"I'm your career advisor. Here's my initial analysis of your resume:{advice}.\n\n Feel free to ask me more about my analysis or other questions that you have. "]

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

            # A hack to "clear" the previous result when submitting a new prompt.
            # from clear_results import with_clear_container

            if submit_clicked:
                res = results_container.container()
                streamlit_handler = StreamlitCallbackHandler(
                    parent_container=res,
                )
                question = prefilled
                session_name = SAMPLE_QUESTIONS[question]
                session_path = Path(__file__).parent / "conv_memory" / session_name
                print(f"Playing saved session: {session_path}")
                response = playback_callbacks(
                    [streamlit_handler], str(session_path), max_pause_time=3
                )
                st.session_state.questions.append(question)
                st.session_state.responses.append(response)

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
                response = new_chat.askAI("1234", question, callbacks=[streamlit_handler])
                st.session_state.questions.append(question)
                st.session_state.responses.append(response)

            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    message(st.session_state['questions'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state['responses'][i], key=str(i))
                




    def initialize(self):

        with placeholder.container():

            with st.form(key='my_form'):

                text_input = st.text_input(
                    "Tell me about your dream job",
                    "",
                    key="job",
                )



                # if text_input:
                #     st.write("Thanks for letting me know!")

            # with col2:
            #     st.text_input(
            #         "Enter some text 👇",
            #         label_visibility=st.session_state.visibility,
            #         disabled=st.session_state.disabled,
            #         placeholder=st.session_state.placeholder,
            #     )

                uploaded_file = st.file_uploader(label="Upload your resume", type=["pdf","odt", "docx","txt"])
                # if uploaded_file is not None:
                #     # To convert to a string based IO:
                #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    # st.write(stringio)

                submit_button = st.form_submit_button(label='Send to resume advisor')
                if submit_button:
                    # loop = asyncio.get_running_loop()
                    # 3. Run in a custom process pool:
                    # with concurrent.futures.ProcessPoolExecutor() as pool:
                    #     result = await loop.run_in_executor(pool, self.assess(uploaded_file, text_input))
                    #     print('custom process pool', result)
                    # queue = Queue()
                    self.id = str(uuid.uuid4())
                    p = Process(target=self.assess, args=(uploaded_file, text_input,))
                    p.start()
                    # subprocess.run([f"{sys.executable}", "assess.py"])
                    self.show_progress()
                    p.join() # this blocks until the process terminates
                    # result = queue.get()
                    # print(result)
                        


    def assess(self, uploaded_file, text_input,):
        file_ext = Path(uploaded_file.name).suffix
        filename = self.id+file_ext
        # uploaded_file.name = filename
        save_path = os.path.join(upload_path, filename)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        read_path =  os.path.join(upload_path, Path(filename).stem+'.txt')
        convert_to_txt(save_path, read_path)
        if (Path(read_path).exists()):
            # Check for content safety
            if (check_content_safety(file=read_path)):
                res_path = os.path.join(result_path, os.path.basename(read_path))
                evaluate_resume(text_input, read_path = read_path, res_path = res_path)

    def show_progress(self):

        with placeholder.container():

            progress_text = "Your career advisor is on its way."
            my_bar = st.progress(0, text=progress_text)
            # to do replaced with real background progress
            for percent_complete in range(100):
                time.sleep(1)
                my_bar.progress(percent_complete + 1, text=progress_text)
                if Path(os.path.join(result_path, self.id+".txt")).is_file():
                    self.create_chatbot()

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
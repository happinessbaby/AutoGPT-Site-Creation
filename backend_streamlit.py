import streamlit as st
from pathlib import Path
import random
import time
import openai
import os
import uuid
from io import StringIO
from langchain.callbacks import StreamlitCallbackHandler
from upgrade_resume import ChatController
from callbacks.capturing_callback_handler import playback_callbacks
from basic_utils import convert_to_txt, check_content_safety, read_txt
from upgrade_resume import basic_upgrade_resume
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

placeholder = st.empty()
upload_path = "./uploads"
result_path = "./static"


class ResumeAdvisor():

    def __init__(self):
        self.advice_file = "./static/advice.txt"


    def create_chatbot(self):

        # st_callback = StreamlitCallbackHandler(st.container())
        new_chat = ChatController(self.advice_file)
        # chat_agent = new_chat.create_chat_agent()

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []


        # Check if 'key' already exists in session_state
        # If not, then initialize it
        # if 'key' not in st.session_state:
        #     st.session_state['key'] = 'value'

        # Set a default model
        # if "openai_model" not in st.session_state:
        #     st.session_state["openai_model"] = "gpt-3.5-turbo"

        st.title("Resume Advisor")

        # expand_new_thoughts = st.sidebar.checkbox(
        #     "Expand New Thoughts",
        #     value=True,
        #     help="True if LLM thoughts should be expanded by default",
        # )


        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

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

        # to be replaced with chat memory
        # SAVED_SESSIONS = {
        #     "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?": "leo.pickle",
        #     "What is the full name of the artist who recently released an album called "
        #     "'The Storm Before the Calm' and are they in the FooBar database? If so, what albums of theirs "
        #     "are in the FooBar database?": "alanis.pickle",
        # }
        SAMPLE_QUESTIONS = {
            "What are some basic steps I can take to improve my resume?": "basic_tips.pickle",
            "What are some things I could be doing terribly wrong with my resume?": "bad_resume.pickle",
        }

        key = "input"
        shadow_key = "_input"


        if key in st.session_state and shadow_key not in st.session_state:
            st.session_state[shadow_key] = st.session_state[key]

        with st.form(key="my_chatbot"):
            prefilled = st.selectbox("Sample questions", sorted(SAMPLE_QUESTIONS.keys())) or ""
            question = st.text_input("Or, ask your own question", key=shadow_key)
            st.session_state[key] = question
            if not question:
                question = prefilled
            submit_clicked = st.form_submit_button("Submit Question")


        question_container = st.empty()
        results_container = st.empty()

        # A hack to "clear" the previous result when submitting a new prompt.
        from clear_results import with_clear_container

        if with_clear_container(submit_clicked):
            # Create our StreamlitCallbackHandler
            res = results_container.container()
            streamlit_handler = StreamlitCallbackHandler(
                parent_container=res,
                # max_thought_containers=int(max_thought_containers),
                # expand_new_thoughts=expand_new_thoughts,
                # collapse_completed_thoughts=collapse_completed_thoughts,
            )

            question_container.write(f"**Question:** {question}")

            # If we've saved this question, play it back instead of actually running LangChain
            # (so that we don't exhaust our API calls unnecessarily)
            if question in SAMPLE_QUESTIONS:
                session_name = SAMPLE_QUESTIONS[question]
                session_path = Path(__file__).parent / "runs" / session_name
                print(f"Playing saved session: {session_path}")
                answer = playback_callbacks(
                    [streamlit_handler], str(session_path), max_pause_time=3
                )
                res.write(f"**Answer:** {answer}")
            else:
                # answer = chat_agent.run(mrkl_input, callbacks=[streamlit_handler])
                answer = new_chat.askAI("1234", question, callbacks=[streamlit_handler])
                res.write(f"**Answer:** {answer}")



    def initialize(self):

        with placeholder.container():

            with st.form(key='my_form'):

                text_input = st.text_input(
                    "Tell me about your dream job",
                    "",
                    key="job",
                )

                if text_input:
                    st.write("Thanks for letting me know!")

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
                    show_progress()
                    file_ext = Path(uploaded_file.name).suffix
                    file_id = str(uuid.uuid4())
                    filename = file_id+file_ext
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
                            self.advice_file = res_path
                            basic_upgrade_resume(text_input, read_path = read_path, res_path = res_path)


def show_progress():
    progress_text = "Your resume advisor is on its way."
    my_bar = st.progress(0, text=progress_text)
    # to do replaced with real background progress
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1, text=progress_text)



if __name__ == '__main__':
    # create_chatbot()
    advisor = ResumeAdvisor()
    # advisor.initialize()
    advisor.create_chatbot()
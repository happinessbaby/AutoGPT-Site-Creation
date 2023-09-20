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
from langchain.agents import AgentType, Tool, load_tools
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
# keyboard = Controller()

# class Keyboard():
#     def on_press(self, key):
#     #print('{0} pressed'.format(
#         #key))
#         if key.char=="space":

#         self.check_key(key)

#     def on_release(self, key):
#         #print('{0} release'.format(
#         # key))
#         if key == Key.esc:
#             # Stop listener
#             return False

#     def check_key(self, key):
#         if key in [Key.up, Key.down, Key.left, Key.right]: 
#             print('YES')
    
# keyboard_event = Keyboard()



class Interview():

    userid: str=""

    def __init__(self):
        self._create_interviewbot()


    def _create_interviewbot(self):

        with placeholder.container():

            with st.sidebar:
                
                add_vertical_space(3)

                st.markdown("How the mock interview works: ")

            if "userid" not in st.session_state:
                st.session_state["userid"] = str(uuid.uuid4())
                print(f"Session: {st.session_state.userid}")
                self.userid = st.session_state.userid

            if "interview_session_id" not in st.session_state:
                st.session_state["interview_session_id"] = str(uuid.uuid4())
                print(f"INTERVIEW Session: {st.session_state.interview_session_id}")
                modal = Modal(title="Welcome to your mock interview session!", key="popup")
                with modal.container():
                    with st.form( key='my_form', clear_on_submit=True):
                        add_vertical_space(1)
                        st.markdown("Please upload your interview material before we begin")
                        st.text_input("links", "", key = "interview_links", )

                        st.file_uploader(label="Upload your files",
                                                        type=["pdf","odt", "docx","txt", "zip", "pptx"], 
                                                        key = "interview_files",
                                                        accept_multiple_files=True)
                        add_vertical_space(1)
                        st.form_submit_button(label='Submit', on_click=self.form_callback)  

                    col1, col2 = st.columns([8, 3])
                    # with col1:
                    #     okay = st.button("okay, I've done so.")
                    with col2:
                        skip = st.button("skip")
                    if skip:
                        # if "baseinterview" not in st.session_state:
                        #     new_interview = InterviewController(st.session_state.userid)
                        #     st.session_state["baseinterview"] = new_interview  
                        st.experimental_rerun()
            

            else:  
                if "baseinterview" not in st.session_state:
                    new_interview = InterviewController(st.session_state.userid)
                    st.session_state["baseinterview"] = new_interview  
                            # st.experimental_rerun() 
                try:
                    self.new_interview = st.session_state.baseinterview
                except AttributeError as e:
                    # if for some reason session ended in the middle, may need to do something different from raise exception
                    raise e
                
                try: 
                    audio_dir =  f"./tmp_recording/{st.session_state.userid}/"
                    os.mkdir(audio_dir)
                except FileExistsError:
                    pass

                
                if 'interview_responses' not in st.session_state:
                    st.session_state['interview_responses'] = list()
                if 'interview_questions' not in st.session_state:
                    st.session_state['interview_questions'] = list()
                    welcome_msg = "Welcome to your mock interview session. I will begin conducting the interview now. "
                    message(welcome_msg, avatar_style="initials", seed="AI_Interviewer", allow_html=True)
                    # self.play_generated_audio(welcome_msg)

                question_container = st.container()
                response_container = st.container()
                
                # while True:
                #     with keyboard.Listener(
                #         on_press=self.on_press,
                #         on_release=self.on_release) as listener:
                #         listener.join()
        
            

                # CODE BELOW IS FOR TESTING W/ TEXT INPUT 
                # hack to clear text after user input
                if 'responseInput' not in st.session_state:
                    st.session_state.responseInput = ''
                def submit():
                    st.session_state.responseInput = st.session_state.interview_input
                    st.session_state.interview_input = ''    
                # User input
                ## Function for taking user provided prompt as input
                def get_text():
                    st.text_input("Your response: ", "", key="interview_input", on_change = submit)
                    return st.session_state.responseInput
                ## Applying the user input box
                with response_container:
                    user_input = get_text()
                    response_container = st.empty()
                    st.session_state.responseInput='' 


                if user_input:
                    res = question_container.container()
                    streamlit_handler = StreamlitCallbackHandler(
                        parent_container=res,
                        # max_thought_containers=int(max_thought_containers),
                        # expand_new_thoughts=expand_new_thoughts,
                        # collapse_completed_thoughts=collapse_completed_thoughts,
                    )
                    user_answer = user_input
                    # answer = chat_agent.run(mrkl_input, callbacks=[streamlit_handler])
                    ai_question = self.new_interview.askAI(user_answer, callbacks = streamlit_handler)
                    st.session_state.interview_questions.append(ai_question)
                    st.session_state.interview_responses.append(user_answer)
                    if st.session_state['interview_responses']:
                        for i in range(len(st.session_state['interview_responses'])-1, -1, -1):
                            message(st.session_state['interview_questions'][i], key=str(i), avatar_style="initials", seed="AI_Interviewer", allow_html=True)
                            message(st.session_state['interview_responses'][i], is_user=True, key=str(i) + '_user',  avatar_style="initials", seed="Yueqi", allow_html=True)

    def on_press(self, key):
        if key==Key.space:
            recorded_audio = self.record_audio(duration, fs, channels)
            user_input = self.transcribe_audio(recorded_audio, fs)
            response = self.new_interview.askAI(user_input)
            self.play_generated_audio(response)

    def on_release(self, key):
        #print('{0} release'.format(             
        # key))
        if key == Key.esc:
            # Stop listener
            return False
        
    def record_audio(self, duration, fs, channels):
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, device=device)
        sd.wait()
        print("Finished recording.")
        return recording

    def transcribe_audio(self, recording, fs):
        filename = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        temp_file = f"./tmp_recording/{self.userid}/{filename}.wav"
        with open(temp_file, "wb") as temp_audio:
        # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_file, recording, fs)
            # sf.write(temp_audio.name, recording, fs)
            temp_audio.close()
            # playsound(temp_audio.name)
            with open(temp_file, "rb") as audio_file:
            # with open(temp_audio.name, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
            # os.remove(temp_audio.name)
        return transcript["text"].strip()
    
    def play_generated_audio(self, text, voice="Bella", model="eleven_monolingual_v1"):
        audio = generate(text=text, voice=voice, model=model)
        play(audio)

    def form_callback(self):

        try: 
            vs_dir=f"./faiss/{st.session_state.userid}/"
            os.mkdir(vs_dir)
        except FileExistsError:
            pass
        try:
            files = st.session_state.interview_files 
            self.process_file(files)
        except Exception:
            pass
        try:
            link = st.session_state.interview_links
            self.process_link(link)
        except Exception:
            pass

        # if "baseinterview" not in st.session_state:
        #     new_interview = InterviewController(st.session_state.userid)
        #     st.session_state["baseinterview"] = new_interview  



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
                # self.add_to_chat(content_type, content_topics, end_path)
                if content_type!="other":
                    name = content_type.strip().replace(" ", "_")
                    vs_name = f"faiss_{name}_{st.session_state.userid}"
                    vs = create_vectorstore("faiss", end_path, "file", vs_name)
                else:
                    vs_name = f"faiss_interview_material_{st.session_state.userid}"
                    vs = merge_faiss_vectorstore(vs_name, end_path)
                vs.save_local(f"./faiss/{st.session_state.userid}/{vs_name}")
            else:
                print("file content unsafe")
                os.remove(end_path)
        

    def process_link(self, posting: Any) -> None:

        """ Processes user shared links including converting all format to txt, checking content safety, and categorizing content type """

        end_path = os.path.join(save_path, st.session_state.userid, str(uuid.uuid4())+".txt")
        if (retrieve_web_content(posting, save_path = end_path)):
            content_safe, content_type, content_topics = check_content(end_path)
            if content_safe:
                if content_type!="other":
                    name = content_type.strip().replace(" ", "_")
                    vs_name = f"faiss_{name}_{str(uuid.uuid4())}"
                    vs = create_vectorstore("faiss", end_path, "file", vs_name)
                else:
                    vs_name = f"faiss_interview_material_{str(uuid.uuid4())}"
                    vs = merge_faiss_vectorstore(vs_name, end_path)
                vs.save_local(f"./faiss/{st.session_state.userid}/{vs_name}")
            else:
                print("link content unsafe")
                os.remove(end_path)


    # def add_to_chat(self, content_type: str, content_topics: set, file_path: str) -> None:

    #     """ Updates entities, vector stores, and tools for different chat agents. The main chat agent will have file paths saved as entities. The interview agent will have files as tools.
        
    #     Args: 

    #         content_type (str): content category, such as resume, cover letter, job posting, other

    #         content_topics (str): if content_type is other, content will be treated as interview material. So in this case, content_topics is interview topics. 

    #         read_path (str): content file path
             
    #      """

    #     if content_type != "other":
    #         name = content_type.strip().replace(" ", "_")
    #         vs_name = f"faiss_{name}_{st.session_state.userid}"
    #         vs = create_vectorstore("faiss", file_path, "file", vs_name)
    #         tool_name = "search_" + vs_name.removeprefix("faiss_").removesuffix(f"_{st.session_state.userid}")
    #         tool_description =  """Useful for generating personalized interview questions and answers. 
    #             Use this tool more than any other tool during a mock interview session when asking personal questions such as work experience, personal projects, tell me about yourself.
    #             Do not use this tool to load any files or documents.  """ 
    #         tools = create_vs_retriever_tools(vs, tool_name, tool_description)
    #         self.new_interview.update_tools(tools)
   
    #     else:
    #         vs_name = f"faiss_interview_material_{st.session_state.userid}"
    #         vs = merge_faiss_vectorstore(vs_name, file_path)
    #         tool_name = "search_" + vs_name.removeprefix("faiss_").removesuffix(f"_{st.session_state.userid}")
    #         tool_description =  """Useful for generating interview questions and answer.
    #             Use this tool more than any other tool during a mock interview session. This tool can also be used for questions about topics in the interview material topics. 
    #             Do not use this tool to load any files or documents. This should be used only for searching and generating questions and answers of the relevant materials and topics. """
    #         tools = create_vs_retriever_tools(vs, tool_name, tool_description)
    #         self.new_interview.update_tools(tools)



if __name__ == '__main__':

    advisor = Interview()
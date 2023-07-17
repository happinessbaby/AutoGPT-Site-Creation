import streamlit as st
from pathlib import Path
import random
import time
import openai
import os
from langchain.callbacks import StreamlitCallbackHandler
from upgrade_resume import ChatController
from callbacks.capturing_callback_handler import playback_callbacks
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

st_callback = StreamlitCallbackHandler(st.container())
new_chat = ChatController()
chat_agent = new_chat.create_chat_agent()

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

st.title("Simple chat")

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

# Going to be replaced with chat memory
SAVED_SESSIONS = {
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?": "leo.pickle",
    "What is the full name of the artist who recently released an album called "
    "'The Storm Before the Calm' and are they in the FooBar database? If so, what albums of theirs "
    "are in the FooBar database?": "alanis.pickle",
}

key = "input"
shadow_key = "_input"


if key in st.session_state and shadow_key not in st.session_state:
    st.session_state[shadow_key] = st.session_state[key]

with st.form(key="form"):
    prefilled = st.selectbox("Sample questions", sorted(SAVED_SESSIONS.keys())) or ""
    mrkl_input = st.text_input("Or, ask your own question", key=shadow_key)
    st.session_state[key] = mrkl_input
    if not mrkl_input:
        mrkl_input = prefilled
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

    question_container.write(f"**Question:** {mrkl_input}")

    # If we've saved this question, play it back instead of actually running LangChain
    # (so that we don't exhaust our API calls unnecessarily)
    if mrkl_input in SAVED_SESSIONS:
        session_name = SAVED_SESSIONS[mrkl_input]
        session_path = Path(__file__).parent / "runs" / session_name
        print(f"Playing saved session: {session_path}")
        answer = playback_callbacks(
            [streamlit_handler], str(session_path), max_pause_time=3
        )
        res.write(f"**Answer:** {answer}")
    else:
        answer = chat_agent.run(mrkl_input, callbacks=[streamlit_handler])
        res.write(f"**Answer:** {answer}")
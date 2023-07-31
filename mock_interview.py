from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.cache import InMemoryCache
import langchain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
langchain.llm_cache = InMemoryCache()







def create_interview_agent():
    messages = [
    SystemMessage(content="You are a helpful assistant that translates write songs."),
    HumanMessage(content="What is faith?")
    ]
    response = chat(messages)
    return response

create_interview_agent()

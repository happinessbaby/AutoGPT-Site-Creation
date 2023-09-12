from langchain.agents import load_tools
from langchain.utilities import TextRequestsWrapper



class JobSearcher():

    def __init__():
        pass

    def get_request(url):
        requests_tools = load_tools(["requests_all"])
        # Each tool wrapps a requests wrapper
        requests_tools[0].requests_wrapper
        TextRequestsWrapper(headers=None, aiosession=None)
        requests = TextRequestsWrapper()
        requests.get(url)

    
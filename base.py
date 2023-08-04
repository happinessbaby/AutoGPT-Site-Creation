class BaseChat():


    def __init__(self):
        pass

    def save_chat(self, chat):
        self.chat = chat

    def get_chat(self):
        print("got chat")
        return self.chat
    

base = BaseChat()
    
    
    


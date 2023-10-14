import geocoder
import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_modal import Modal
import json
from st_pages import show_pages_from_config, add_page_title, show_pages, Page
from st_clickable_images import clickable_images
import base64
from langchain.agents import load_tools
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
placeholder = st.empty()

class Resources():

    def __init__(self):

        pass
        # self.set_clickable_icons()
        # self._create_resources()

    @st.cache_data(experimental_allow_widgets=True)
    def set_clickable_icons(_self):

        images = []
        for file in ["image1.png", "image2.png"]:
            with open(file, "rb") as image:
                encoded = base64.b64encode(image.read()).decode()
                images.append(f"data:image/png;base64,{encoded}")
        clicked = clickable_images(
            images,
            titles=[f"Image #{str(i)}" for i in range(2)],
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )
        # if "clicked_job_fair" not in st.session_state:
        #     st.session_state["clicked_job_fair"] = clicked_job_fair
        # clicked_networking = clickable_images(
        #     [images[1]],
        #     titles = "Social Networking",
        #     div_style = {"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        #     img_style={"margin": "5px", "height": "200px"},
        # )
        # if "clicked_networking" not in st.session_state:
        #     st.session_state["clicked_networking"] = clicked_networking
        # if st.session_state.clicked_job_fair
        if clicked:
            _self.find_job_fairs()

            
    def find_job_fairs(self):

        # TODO Job fair events
        print("inside search job fairs")
        # ask user to use their current location or somewhere else
        location = self.get_location()



    
    def find_social_networking(self):
        
        # TODO LinkedIn Events
        return None

    def get_location(self):

        g = geocoder.ip('me')
        print(g.latlng)





if __name__== '__main__':
    resources = Resources()


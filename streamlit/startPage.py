from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from urllib.error import URLError
import os


try:

    # Set page layout to wide screen and menu item
    menu_items = {
	'Get help': None,
	'Report a bug': None,
	'About': "Rag With Image Docs"
    }
    st.set_page_config(layout="wide", menu_items=menu_items)

    col1, col2, col3 = st.columns([1,2,2])
    with col1:
        st.image(os.path.join('images','microsoft.png'))

    col1, col2, col3 = st.columns([2,2,1])
    with col3:
        model = st.selectbox(
            "Type", ["","Text", "Image"]
        )

    question = st.text_input("AI Assistant Ready", value="Select menue item at right hand side")


except URLError as e:
    st.error(
        """
        **Error in Demo**
        error: %s
        """
        % e.reason
    )
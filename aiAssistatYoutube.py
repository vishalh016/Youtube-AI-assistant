# This code snippet is for webapp in streamlit app which is basically youtube assistant 
# to run this type following into the terminal:
# streamlit run ai
#
#


import streamlit as st
import langchainhelper
import textwrap

st.title("Youtube Assistant")
with st.sidebar:
    with st.form(key='myform'):
        youtube_url = st.sidebar.text_area(
            label="Type your Youtube video url please:",
            max_chars=100

        )
        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=500,
            key="query"
        )
        submit_button = st.form_submit_button(label='Submit')
if query and youtube_url:
    db = langchainhelper.create_vectorDB_from_youtube_url(youtube_url)
    response, docs = langchainhelper.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))

# Bishal

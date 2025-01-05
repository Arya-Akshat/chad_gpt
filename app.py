import streamlit as st
import time
import google.generativeai as genai
import os
import pandas as pd

template = ["You are an expert at refining prompts for large language models. ",
                "Analyze the following prompt given by the user:\n\n",
                "Prompt: {user_prompt}\n\n",
                "Provide feedback on how to make this prompt more effective, along with ",
                "examples of improved versions. Consider:\n",
                "1. Clarity and specificity\n",
                "2. Context and constraints\n",
                "3. Format and structure\n",
                "4. Potential ambiguities\n"]

api_key = 'AIzaSyB47MN1x_-5ZUUBgI-qEaqybb4aysvQ_TM'
genai.configure(api_key=str(API))
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=''.join(template))

st.title("Chad GPT")

def response_generator(user_prompt):
    response = model.generate_content(user_prompt, stream=True)
    for word in response:
        yield word.text
        time.sleep(0.05)

if 'messages' not in st.session_state:
    st.session_state.messages = []   #{role: , content: }

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

input = st.chat_input('Enter your prompt')
if input:
    with st.chat_message('user'):
        st.write(input)
        st.session_state.messages.append({'role': 'user', 'content': input})

    with st.chat_message('ai'):
        response = st.write_stream(response_generator(input))
        st.session_state.messages.append({'role': 'ai', 'content': response})

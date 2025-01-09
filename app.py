import streamlit as st
import time
import google.generativeai as genai
import os
import pandas as pd

template = ["You are an expert in crafting detailed, accurate, and user-friendly responses for laboratory experiments.\n"  
                    "Analyze the following experiment-related prompt:\n\n"  
                    "Prompt: {user_prompt}\n\n"  
                    "Provide a comprehensive response covering the following aspects:\n"  
                    "1. Objective: define the purpose of the experiment.\n"  
                    "2. Components and Materials Required: Provide  list of all components, equipment, and materials needed.\n"  
                    "3. Step-by-Step Procedure: Outline the procedure in clear, sequential steps, ensuring it is easy to follow for the intended audience.\n"  
                    "5. Source Materials: Suggest reliable online resourcess.\n"  
                      
                     
                    "incase of invalid details shared, ask the user to reframe their words"]

API = 'AIzaSyB47MN1x_-5ZUUBgI-qEaqybb4aysvQ_TM'
genai.configure(api_key=str(API))
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=''.join(template))

st.title("PromptForge - prompt helper")

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

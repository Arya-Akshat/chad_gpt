import streamlit as st
import time
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
import google.generativeai as genai


class GeminiLLM(LLM, BaseModel):

    api_key: str = Field(..., description="API key for Gemini")
    model_name: str = Field(default="gemini-pro", description="Gemini model name")
    _model: Any = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = self._model.generate_content(prompt)
            if hasattr(response, 'text'):
                return response.text
            else:
                return str(response)
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}


class PromptEvaluator:

    def __init__(self, api_key: str):
        self.llm = GeminiLLM(api_key=api_key)
        self.evaluate_template = PromptTemplate(
            input_variables=["user_prompt"],
            template=(
                "You are an expert at refining prompts for large language models. "
                "Analyze the following prompt given by the user:\n\n"
                "Prompt: {user_prompt}\n\n"
                "Provide feedback on how to make this prompt more effective, along with "
                "examples of improved versions. Consider:\n"
                "1. Clarity and specificity\n"
                "2. Context and constraints\n"
                "3. Format and structure\n"
                "4. Potential ambiguities\n"
            )
        )
        self.evaluate_chain = self.evaluate_template | self.llm

    def get_prompt_suggestions(self, user_prompt: str) -> str:
        try:
            return self.evaluate_chain.invoke({"user_prompt": user_prompt})
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"


# Streamlit App
st.title("PromptForge: Elevate Your Prompt Crafting")
st.sidebar.header("About")
st.sidebar.write("PromptForge is a cutting-edge tool designed to help users craft highly effective prompts for large language models. Whether you're an AI enthusiast, developer, or researcher, this tool empowers you to:")

# Initialize PromptEvaluator with API key
API_KEY = "AIzaSyB47MN1x_-5ZUUBgI-qEaqybb4aysvQ_TM"
evaluator = PromptEvaluator(api_key=API_KEY)

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

input = st.chat_input('Enter your prompt')
if input:
    with st.chat_message('user'):
        st.write(input)
        st.session_state.messages.append({'role': 'user', 'content': input})

    with st.chat_message('ai'):
        with st.spinner("Analyzing your prompt..."):
            response = evaluator.get_prompt_suggestions(input)
        st.write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})

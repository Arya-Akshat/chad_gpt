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
                                    "You are an expert in crafting detailed, accurate, and user-friendly responses for laboratory experiments.\n"  
                    "Analyze the following experiment-related prompt:\n\n"  
                    "Prompt: {user_prompt}\n\n"  
                    "Provide a comprehensive response covering the following aspects:\n"  
                    "1. Objective: Clearly define the purpose of the experiment.\n"  
                    "2. Components and Materials Required: Provide a detailed list of all components, equipment, and materials needed, including specifications where applicable.\n"  
                    "3. Step-by-Step Procedure: Outline the procedure in clear, sequential steps, ensuring it is easy to follow for the intended audience.\n"  
                    "4. Safety and Security Measures: List all safety precautions and protocols to follow during the experiment to ensure safety in the laboratory. Also mentions hazards involved and BSL level.\n"  
                    "5. How to Use Links and Source Materials: Suggest reliable online resources or references for understanding key concepts, sourcing materials, or troubleshooting issues.\n"  
                    "6. Additional Notes: Include tips, best practices, or potential challenges to consider during the experiment.\n\n"  
                    "Response Example:\n"  
                    "Based on the query, generate a structured response with headings like:\n"  
                    "- Objective\n"  
                    "- Materials Required\n"  
                    "- Procedure\n"  
                    "- Safety Precautions\n"  
                    "- Additional Notes\n"  
                    "incase of invalid details shared, ask the user to reframe their words"

            )
        )
        self.evaluate_chain = self.evaluate_template | self.llm

    def get_prompt_suggestions(self, user_prompt: str) -> str:
        try:
            return self.evaluate_chain.invoke({"user_prompt": user_prompt})
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"


# Streamlit App
st.title("Lab-Pro: Your Daily Lab Assistant ")
st.sidebar.header("About")
st.sidebar.write("Lab-Pro is an advanced chatbot designed to provide comprehensive guidance for laboratory experiments. Whether you're a student, researcher, or educator, Lab-Pro offers detailed step-by-step procedures, safety measures, materials, and links to trusted resources. Its intuitive interface ensures clear and accurate responses for a wide range of scientific experiments, making it an essential tool for enhancing laboratory learning and research.")

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

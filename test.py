from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
import google.generativeai as genai


class GeminiLLM(LLM, BaseModel):
    """Custom LLM class for integrating Gemini with LangChain"""

    api_key: str = Field(..., description="API key for Gemini")
    model_name: str = Field(default="gemini-pro", description="Gemini model name")
    # Use PrivateAttr for the model instance
    _model: Any = PrivateAttr()

    class Config:
        """Pydantic model configuration"""
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        # Initialize the model as a private attribute
        self._model = genai.GenerativeModel(self.model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Make a call to the Gemini API"""
        try:
            # Generate content using Gemini
            response = self._model.generate_content(prompt)
            # Check if the response has text
            if hasattr(response, 'text'):
                return response.text
            else:
                return str(response)
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM"""
        return "gemini"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for this LLM instance"""
        return {"model_name": self.model_name}


class PromptEvaluator:
    """Class for evaluating and improving prompts using Gemini"""

    def __init__(self, api_key: str):
        """Initialize with API key"""
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
        # Create a runnable sequence
        self.evaluate_chain = self.evaluate_template | self.llm

    def get_prompt_suggestions(self, user_prompt: str) -> str:
        """Get suggestions for improving a prompt"""
        try:
            # Invoke the runnable sequence with the input
            return self.evaluate_chain.invoke({"user_prompt": user_prompt})
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"


def main():
    """Main function to demonstrate usage"""
    try:
        # You'll need to get an API key from https://makersuite.google.com/app/apikey
        api_key = 'AIzaSyB47MN1x_-5ZUUBgI-qEaqybb4aysvQ_TM'

        evaluator = PromptEvaluator(api_key=api_key)

        print("Lab-PRO: Your daily lab Assistant")
        print("------------------------------------------------")
        print("Type 'quit' to exit the program.\n")

        while True:
            user_input = input("\nEnter a prompt to improve: ")
            if user_input.lower() == 'quit':
                break

            print("\nAnalyzing your prompt...")
            suggestions = evaluator.get_prompt_suggestions(user_input)
            print("\nSuggestions for prompt improvement:")
            print(suggestions)

    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    main()

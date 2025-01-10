import time
import google.generativeai as genai
import os

# Define the prompt template
template = ("You are an expert in crafting detailed, accurate, and user-friendly responses for laboratory experiments.\n"
            "Analyze the following experiment-related prompt:\n\n"
            "Prompt: {user_prompt}\n\n"
            "Provide a comprehensive response covering the following aspects:\n"
            "1. Objective: Define the purpose of the experiment.\n"
            "2. Components and Materials Required: Provide a list of all components, equipment, and materials needed.\n"
            "3. Step-by-Step Procedure: Outline the procedure in clear, sequential steps, ensuring it is easy to follow for the intended audience.\n"
            "4. Source Materials: Suggest reliable online resources.\n"
            "In case of invalid details shared, ask the user to reframe their words.")

# Configure the GenAI API
API = 'AIzaSyB47MN1x_-5ZUUBgI-qEaqybb4aysvQ_TM'
genai.configure(api_key=API)

# Define the generative model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=template
)

def response_generator(user_prompt):
    response = model.generate_content(user_prompt, stream=True)
    for word in response:
        yield word.text
        time.sleep(0.05)

def main():
    print("Welcome to PromptForge - Command Line Edition")
    print("Type your experiment prompt below. Type 'exit' to quit.")
    
    while True:
        user_prompt = input("\nEnter your prompt: ").strip()
        
        if user_prompt.lower() == 'exit':
            print("Goodbye!")
            break
        
        print("\nGenerating response...")
        response = []
        for word in response_generator(user_prompt):
            print(word, end='', flush=True)
            response.append(word)
        print("\n")
        print("Response generated successfully!")

if __name__ == "__main__":
    main()

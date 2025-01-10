import time
import google.generativeai as genai

# Define the template for refining prompts
template = [
    "You are an expert at refining prompts for large language models. ",
    "Analyze the following prompt given by the user:\n\n",
    "Prompt: {user_prompt}\n\n",
    "Provide feedback on how to make this prompt more effective, along with ",
    "examples of improved versions. Consider:\n",
    "1. Clarity and specificity\n",
    "2. Context and constraints\n",
    "3. Format and structure\n",
    "4. Potential ambiguities\n"
]

# Configure the Gemini API
API = "YOUR_API_KEY_HERE"  # Replace with your API key
genai.configure(api_key=API)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=''.join(template)
)

# Function to generate a response
def response_generator(user_prompt):
    """Generate a response from the Gemini model."""
    try:
        print("\nGenerating response...")  # Debugging message
        response = model.generate_content(user_prompt, stream=True)
        full_response = ""
        for word in response:
            full_response += word.text
            time.sleep(0.05)  # Optional for streaming effect
        return full_response
    except Exception as e:
        return f"Error while generating response: {str(e)}"

# Main CLI loop
def main():
    """Main function to handle user interaction via CLI."""
    print("Welcome to Prompt Refinement Tool (CLI Version)")
    print("-------------------------------------------------")
    print("Type 'exit' to quit.\n")

    while True:
        # Get user input
        user_input = input("Enter your prompt: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Generate and display response
        response = response_generator(user_input)
        print("\nSuggestions for improving your prompt:")
        print(response)
        print("\n-------------------------------------------------\n")

# Run the program
if __name__ == "__main__":
    main()

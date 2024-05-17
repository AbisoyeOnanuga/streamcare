import replicate
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()   

# Retrieve the API token from the environment variable
REPLICATE_API_KEY = os.getenv('REPLICATE_API_KEY')

# Initialize the Replicate model with the API key
def initialize_model():
    return replicate.Client(api_token=REPLICATE_API_KEY)

# Function to get a response from the snowflake-arctic-instruct model
def get_model_response(question):
    input = {
        "prompt": question,
        "temperature": 0.2
    }
    
    try:
        client = initialize_model()
        output = ""
        for event in client.stream(
            "snowflake/snowflake-arctic-instruct",
            input=input
        ):
            if 'output' in event:
                output += event['output']
        return output
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app layout
st.title('AI Response Generator')

# Input field for the question
question = st.text_input('Ask a question to the AI model')

# When the 'Get Response' button is pressed
if st.button('Get Response'):
    response = get_model_response(question)
    
    # Display the response
    if response:
        st.text_area('Response', response, height=300)
    else:
        st.error("No response generated or an error occurred.")

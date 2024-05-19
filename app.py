import replicate
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Initialize the Replicate model with the API key
def initialize_model():
    return replicate.Client(api_token=REPLICATE_API_TOKEN)

# Function to get a response from the snowflake-arctic-instruct model
def get_model_response(medications, side_effects, medical_condition):
    input = {
        "prompt": (
            f"As an AI trained in pharmacology, analyze the potential drug-related causes of side effects and the "
            f"impact of the patient's medical condition on their treatment. Medications: {medications}. Reported "
            f"side effects: {side_effects}. Medical condition: {medical_condition}. Provide detailed insights that "
            f"can aid healthcare professionals in making informed treatment decisions."
        ),
        "temperature": 0.2
    }
    
    st.write("Sending the following input to the model:", input)  # Debug print
    
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
        st.error("Exception type: " + str(type(e)))
        st.error("Exception args: " + str(e.args))
        return None

# Streamlit app layout
st.title('Patient Data Analysis')

# Input fields for the model
medications = st.text_input('List of medications')
side_effects = st.text_area('Description of side effects')
medical_condition = st.text_input('Medical condition')

# When the 'Analyze' button is pressed
if st.button('Analyze'):
    response = get_model_response(medications, side_effects, medical_condition)
    
    # Display the response
    if response:
        st.write(response)
    else:
        st.error("No response generated or an error occurred.")
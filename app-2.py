import os
import replicate
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Initialize the Replicate model with the API key
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Streamlit app layout
st.title('Patient Data Analysis')

# Input fields for the model
medications = st.text_input('List of medications')
side_effects = st.text_area('Description of side effects')
medical_condition = st.text_input('Medical condition')

# When the 'Analyze' button is pressed
if st.button('Analyze'):
    # Construct the prompt using the user inputs
    prompt = (
        f"As an AI trained in pharmacology, analyze the potential drug-related causes of side effects and the "
        f"impact of the patient's medical condition on their treatment. Medications: {medications}. Reported "
        f"side effects: {side_effects}. Medical condition: {medical_condition}. Provide detailed insights that "
        f"can aid healthcare professionals in making informed treatment decisions."
    )
    
    # Set up the input dictionary for the Replicate API call
    input = {
        "prompt": prompt,
        "temperature": 0.2
    }
    
    # Stream the response from the model
    output = ""
    try:
        for event in client.stream(
            "snowflake/snowflake-arctic-instruct",
            input=input
        ):
            if 'output' in event:
                output += event['output']
                st.write(event)  # Debug print
        # Display the response in the Streamlit app
        if output:
            st.text_area('Response', output, height=300)
        else:
            st.error("The model returned an empty response.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
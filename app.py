import replicate
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Replicate API key
REPLICATE_API_KEY = os.getenv('REPLICATE_API_TOKEN')  # Ensure this is set in your .env file

# Initialize the Replicate model with the API key
def initialize_model():
    return replicate.Client(api_token=REPLICATE_API_KEY)

# Function to analyze patient data using the snowflake-arctic-instruct model
def analyze_patient_data(prompt):
    input = {
        "prompt": prompt,
        "temperature": 0.2,
        # Add other parameters as needed
    }
    
    try:
        client = initialize_model()
        for event in client.stream(
            "snowflake/snowflake-arctic-instruct",
            input=input
        ):
            if 'output' in event:
                return event['output']
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app layout
st.title('Streamcare: AI ✨+ Data 📊 personalised care')

# Input fields for patient data
with st.form(key='patient_form'):
    side_effects_notes = st.text_area('Tell us about your side effects')
    medication_list = st.text_input('What current medication are you taking')
    medical_condition = st.text_input('Tell us about your medical condition')
    submit_button = st.form_submit_button(label='Submit')

# When the submit button is pressed
if submit_button:
    # Construct the prompt for the Replicate model
    prompt = (
        "As an AI trained in pharmacology, analyze the potential drug-related causes of side effects "
        "and the impact of the patient's medical condition on their treatment. "
        f"Medications: {medication_list}. Reported side effects: {side_effects_notes}. "
        f"Medical condition: {medical_condition}. "
        "Provide detailed insights that can aid healthcare professionals in making informed treatment decisions."
    )
    
    markdown_results = analyze_patient_data(prompt)
    
    # Display the results
    if markdown_results:
        st.markdown(markdown_results, unsafe_allow_html=True)
    else:
        st.error("No response generated.")
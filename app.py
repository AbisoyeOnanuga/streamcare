import streamlit as st
from dotenv import load_dotenv
import os
import requests

# Load the environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Define a function to call the Gemini API
def call_gemini_api(patient_data):
    # Replace 'your_gemini_api_endpoint' with the actual endpoint
    url = 'your_gemini_api_endpoint'
    headers = {
        'Authorization': f'Bearer {GEMINI_API_KEY}',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json=patient_data)
    return response.json()

# Streamlit app layout
st.title('Personalized Primary Care App')

# Input fields for patient data
with st.form(key='patient_form'):
    side_effects_notes = st.text_area('Side Effects Notes')
    medication_list = st.text_input('Medication List')
    medical_condition = st.text_input('Medical Condition')
    submit_button = st.form_submit_button(label='Submit')

# When the submit button is pressed
if submit_button:
    patient_data = {
        'side_effects_notes': side_effects_notes,
        'medication_list': medication_list,
        'medical_condition': medical_condition
    }
    
    # Call the Gemini API or any other AI model API
    results = call_gemini_api(patient_data)
    
    # Display the results
    st.write('Results:')
    st.json(results)

# Modular AI model API endpoint
AI_MODEL_ENDPOINT = os.getenv('AI_MODEL_ENDPOINT', 'your_default_ai_api_endpoint')

# You can switch the AI model by changing the environment variable
# For example, in your .env file, you can have:
# AI_MODEL_ENDPOINT=your_other_ai_api_endpoint

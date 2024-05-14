import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Configure the Google API key
GEMINI_API_KEY = os.getenv('GEMINI_KEY')  # Ensure this is set in your .env file

# Initialize the Gemini model with the API key
def initialize_model():
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        'gemini-pro',
        generation_config=genai.GenerationConfig(
            max_output_tokens=2000,
            temperature=0.9,
        )
    )
    return model

def analyze_patient_data(medication_list, side_effects_notes, medical_condition):
    prompt = (
        "As an AI trained in pharmacology, analyze the potential drug-related causes of side effects "
        "and the impact of the patient's medical condition on their treatment. "
        f"Medications: {medication_list}. Reported side effects: {side_effects_notes}. "
        f"Medical condition: {medical_condition}. "
        "Provide detailed insights that can aid healthcare professionals in making informed treatment decisions."
    )
    try:
        model = initialize_model()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app layout
st.title('Streamcare: AI + Data personalised care')

# Input fields for patient data
with st.form(key='patient_form'):
    side_effects_notes = st.text_area('Tell us about your side effects')
    medication_list = st.text_input('What current medication are you taking')
    medical_condition = st.text_input('Tell us about your medical condition')
    submit_button = st.form_submit_button(label='Submit')

# When the submit button is pressed
if submit_button:
    markdown_results = analyze_patient_data(medication_list, side_effects_notes, medical_condition)
    
    # Display the results
    if markdown_results:
        # Check if the result is not empty and is a valid JSON string
        if markdown_results.strip() and markdown_results.startswith('{') and markdown_results.endswith('}'):
            try:
                # Attempt to parse the JSON response
                parsed_results = json.loads(markdown_results)
                st.write('Results:')
                st.json(parsed_results)
            except json.JSONDecodeError as e:
                # Log the error and the problematic data
                st.error(f"JSON parsing error: {e}")
                st.text("Received data:")
                st.write(markdown_results)
        else:
            # If the result is not JSON, render as Markdown
            st.markdown(markdown_results, unsafe_allow_html=True)
    else:
        st.error("No response generated.")

# You can switch the AI model by changing the environment variable
# For example, in your .env file, you can have:
# AI_MODEL_ENDPOINT=your_other_ai_api_endpoint

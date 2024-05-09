import os
import google.generativeai as genai
from dotenv import load_dotenv

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

# Sample cancer medication list for a given patient (replace with actual data)
medication_list = [
    {"name": "Tamoxifen", "dosage": "20mg", "frequency": "Once a day"},
    {"name": "Methotrexate", "dosage": "2.5mg", "frequency": "Once a week"},
    {"name": "Cisplatin", "dosage": "75mg/m2", "frequency": "Every 3 weeks"}
]

# Sample side effects note (replace with actual data)
side_effects_note = "Patient reports experiencing mild nausea and occasional dizziness, particularly after the last chemotherapy session."

# Craft a detailed prompt for analyzing the patient's medication list and side effects
prompt = (
    "As an AI trained in pharmacology, analyze the potential drug-related causes of side effects based on the patient's medication list. "
    f"Medications: {medication_list}. Reported side effects: {side_effects_note}. "
    "Provide detailed insights that can aid healthcare professionals in making informed treatment decisions."
)

# Function to analyze the patient's medication list and side effects
def analyze_patient_note(prompt):
    try:
        model = initialize_model()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
response = analyze_patient_note(prompt)
if response:
    print(response)
else:
    print("No response generated.")
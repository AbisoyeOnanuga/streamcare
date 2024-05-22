import os
import replicate
import streamlit as st
from dotenv import load_dotenv
from utils import generate_synthetic_data, training_log_performance, stream_with_retries

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Initialize the Replicate model with the API key
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

def run_training_simulation(client, model_name, num_cases, st):
    synthetic_cases = generate_synthetic_data(num_cases)
    global test_count
    test_count = 0  # Initialize test count

    for case in synthetic_cases:
        test_count += 1  # Increment test count for each case
        st.write("Simulated Patient Scenario:", case)

        # Streamlit UI for user input
        user_diagnosis = st.text_input("Enter your diagnosis:", key=f"diagnosis_{test_count}")
        user_treatment_plan = st.text_input("Enter your suggested treatment plan:", key=f"treatment_{test_count}")

        # Construct the prompt for AI feedback
        ai_feedback_prompt = (
            f"User Diagnosis: {user_diagnosis}\n"
            f"User Treatment Plan: {user_treatment_plan}\n"
            f"Please provide feedback and additional insights based on the user's input and the following patient information:\n"
            f"Medications: {case['medications']}\n"
            f"Reported Side Effects: {case['side_effects']}\n"
            f"Medical Condition: {case['medical_condition']}\n"
        )

        # Use the retry function to handle model streaming
        ai_feedback_outputs = list(stream_with_retries(model_name, {'prompt': ai_feedback_prompt, 'temperature': 0.2}))

        # Combine all parts of the AI feedback into one string
        ai_feedback = ' '.join(ai_feedback_outputs)

        # Display AI feedback in Streamlit
        st.write("AI Feedback:", ai_feedback)

        # Log the performance for the current case
        training_log_performance('Training-simulation', model_name, case, ai_feedback, test_count)

        # Ensure that the next case has fresh input fields
        st.write("---")  # Visual separator for the next case
        
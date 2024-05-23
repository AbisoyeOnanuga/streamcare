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

model_name = "snowflake/snowflake-arctic-instruct"

def run_training_simulation(client, model_name, num_cases, st):
    synthetic_cases = generate_synthetic_data(num_cases)
    global test_count
    test_count = 0  # Initialize test count

    # Limit the number of cases to 10
    synthetic_cases = synthetic_cases[:10]

    for case_index, case in enumerate(synthetic_cases):
        test_count += 1  # Increment test count for each case
        st.write("Simulated Patient Scenario:", case)

        # Initialize session state for user inputs if not already present
        diagnosis_key = f"diagnosis_{case_index}"
        treatment_key = f"treatment_{case_index}"
        feedback_key = f"ai_feedback_{case_index}"

        if diagnosis_key not in st.session_state:
            st.session_state[diagnosis_key] = ""
        if treatment_key not in st.session_state:
            st.session_state[treatment_key] = ""
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = ""

        # Display input fields with values from session state
        user_diagnosis = st.text_input("Enter your diagnosis:", value=st.session_state[diagnosis_key], key=diagnosis_key)
        user_treatment_plan = st.text_input("Enter your suggested treatment plan:", value=st.session_state[treatment_key], key=treatment_key)

        # Button to submit the diagnosis and treatment plan
        if st.button('Submit Diagnosis', key=f"submit_{case_index}"):
            st.session_state[feedback_key] = generate_ai_feedback(case, user_diagnosis, user_treatment_plan)

        # Display AI feedback from session state
        if st.session_state[feedback_key]:
            st.markdown("AI Feedback:")
            st.markdown(st.session_state[feedback_key])

        st.write("---")  # Visual separator for the next case

def generate_ai_feedback(case, user_diagnosis, user_treatment_plan):
    # Construct the prompt for AI feedback
    ai_feedback_prompt = construct_ai_prompt(case, user_diagnosis, user_treatment_plan)

    # Generate AI feedback using the retry function to handle model streaming
    ai_feedback_outputs = list(stream_with_retries(model_name, {'prompt': ai_feedback_prompt, 'temperature': 0.2}))
    ai_feedback = ' '.join(ai_feedback_outputs).replace('  ', ' ')  # Remove extra spaces

    # Log the performance for the current case
    training_log_performance('Training-simulation', model_name, case, ai_feedback, test_count)

    return ai_feedback

def construct_ai_prompt(case, user_diagnosis, user_treatment_plan):
    # Construct and return the AI feedback prompt
    ai_feedback_prompt = (
        f"Based on the patient information provided, analyze the potential causes of the reported side effects "
        f"and the impact of the medical condition on the treatment. Then, provide feedback on the user's diagnosis "
        f"and treatment plan.\n"
        f"- **Medications**: {case['medications']}\n"
        f"- **Reported Side Effects**: {case['side_effects']}\n"
        f"- **Medical Condition**: {case['medical_condition']}\n\n"
        f"**Actionable Steps**:\n"
        f"- Succinctly suggest adjustment to the **dosage of {case['medications']}**.\n"
        f"- Succinctly suggest a **monitoring duration** based on {case['side_effects']} and {case['medical_condition']}.\n\n"
        f"- Succinctly suggest any potential adjustments to the medications treatment plan.\n"
        f"Please review these suggestions with a healthcare professional."
        f"generate a response following the above pattern in markdown"
    )
    return ai_feedback_prompt

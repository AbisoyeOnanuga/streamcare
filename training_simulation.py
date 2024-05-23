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

def run_training_simulation(num_cases, st, model_name):
    synthetic_cases = generate_synthetic_data(num_cases)
    test_type = 'Synthetic'
    global test_count
    test_count = 0

    # Limit the number of cases to 10
    synthetic_cases = synthetic_cases[:10]

    for case_index, case in enumerate(synthetic_cases):
        test_count += 1  # Increment test count for each case
        input_data = {
            'medications': case['medications'],
            'side_effects': case['side_effects'],
            'medical_condition': case['medical_condition']
        }
        input_prompt = (
            f"### Synthetic Test Analysis\n"
            f"**Medications**: {case['medications']}\n"
            f"**Reported Side Effects**: {case['side_effects']}\n"
            f"**Medical Condition**: {case['medical_condition']}\n\n"
            f"**Statistical Analysis**:\n"
            f"- There is a **[analyse probability]%** likelihood that the side effects are related to **{case['medications']}**.\n\n"
            f"**Actionable Steps**:\n"
            f"- Succinctly suggest adjustment to the **dosage of {case['medications']}**.\n"
            f"- Succinctly suggest a **monitoring duration** based on {case['side_effects']} and {case['medical_condition']}.\n\n"
            f"- Succinctly suggest any potential adjustments to the medications treatment plan.\n"
            f"Please review these suggestions with a healthcare professional."
            f"generate a response following the above pattern in markdown"
        )
        # Use the retry function to handle model streaming
        model_outputs = list(stream_with_retries(model_name, {'prompt': input_prompt, 'temperature': 0.2}))
        ai_feedback = ' '.join(model_outputs).replace('  ', ' ')  # Remove extra spaces

        # Display the synthetic case and AI feedback in Streamlit
        st.json(case)  # Display the case as JSON
        st.markdown("AI Feedback:")
        st.markdown(ai_feedback)  # Display the AI feedback

        # Log the performance for the current case
        training_log_performance(test_type, model_name, input_data, model_outputs, test_count)

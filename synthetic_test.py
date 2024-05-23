import os
import replicate
from dotenv import load_dotenv
from utils import generate_synthetic_data, log_performance, stream_with_retries

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Initialize the Replicate model with the API key
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

model_name = "snowflake/snowflake-arctic-instruct"

def run_synthetic_test(num_cases, st):
    synthetic_cases = generate_synthetic_data(num_cases)
    test_type = 'Synthetic'
    global test_count
    test_count = 0
    for case in synthetic_cases:
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
            f"**AI-Insights**:\n"
            f"Generate a statistical analysis of the likelihood of the side effects being related to the medications "
            f"**Actionable Steps**:\n"
            f"and suggest any potential adjustments to the medications treatment plan."
        )
        # Use the retry function to handle model streaming
        model_outputs = list(stream_with_retries(model_name, {'prompt': input_prompt, 'temperature': 0.2}))
        ai_feedback = ' '.join(model_outputs).replace('  ', ' ')  # Remove extra spaces
        st.session_state.ai_responses.append(ai_feedback)  # Store AI feedback in session state

        # Display the synthetic case and AI feedback in Streamlit
        st.json(case)  # Display the case as JSON
        st.markdown("AI Feedback:")
        st.markdown(ai_feedback)  # Display the AI feedback        
        # Log the performance for the current case
        log_performance(test_type, model_name, input_data, model_outputs, test_count)
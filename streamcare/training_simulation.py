import os
import replicate
from dotenv import load_dotenv
from utils import generate_synthetic_data, user_diagnosis_and_treatment, ai_feedback_on_user_input, training_log_performance, stream_with_retries

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Initialize the Replicate model with the API key
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

def run_training_simulation(model_name, num_cases):
    synthetic_cases = generate_synthetic_data(num_cases)
    global test_count
    for case in synthetic_cases:
        test_count += 1  # Increment test count for each case
        print("Simulated Patient Scenario:", case)
        user_diagnosis, user_treatment_plan = user_diagnosis_and_treatment()
        ai_feedback = ai_feedback_on_user_input(user_diagnosis, user_treatment_plan, case)
        print("AI Feedback:", ai_feedback)
        # Log the performance for the current case
        training_log_performance('Training-simulation', model_name, case, ai_feedback, test_count)
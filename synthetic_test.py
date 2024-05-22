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

def run_synthetic_test(num_cases):
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
            f"As an AI trained in pharmacology, analyze the potential drug-related causes of side effects and the "
            f"impact of the patient's medical condition on their treatment. Medications: {case['medications']}. Reported "
            f"side effects: {case['side_effects']}. Medical condition: {case['medical_condition']}. Provide detailed insights that "
            f"can aid healthcare professionals in making informed treatment decisions."
        )
        # Use the retry function to handle model streaming
        model_outputs = list(stream_with_retries(model_name, {'prompt': input_prompt, 'temperature': 0.2}))
        # Log the performance for the current case
        log_performance(test_type, model_name, input_data, model_outputs, test_count)
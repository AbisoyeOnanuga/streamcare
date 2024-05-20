import os
import replicate
from dotenv import load_dotenv
import logging
from datetime import datetime
import random

# Configure logging to capture both console output and file output
logging.basicConfig(filename='model_performance.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize test count
test_count = 0

# Function to log performance with timestamp and test iteration
def log_performance(input_data, model_output, test_count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'Test #{test_count} | Timestamp: {timestamp} | Input: {input_data} | Output: {model_output}')

# Extended lists for synthetic data generation
medications_list = ['Aspirin', 'Metformin', 'Lisinopril', 'Atorvastatin', 'Simvastatin', 'Levothyroxine']
side_effects_list = ['Nausea', 'Dizziness', 'Headache', 'Fatigue', 'Dry mouth', 'Insomnia']
medical_conditions_list = ['Type 2 Diabetes', 'Hypertension', 'High Cholesterol', 'Hypothyroidism', 'Asthma']

# Function to generate synthetic data
def generate_synthetic_data(num_cases=10):
    synthetic_data = []
    for _ in range(num_cases):
        combo = {
            'medications': ', '.join(random.sample(medications_list, random.randint(1, 3))),
            'side_effects': ', '.join(random.sample(side_effects_list, random.randint(1, 3))),
            'medical_condition': random.choice(medical_conditions_list)
        }
        synthetic_data.append(combo)
    return synthetic_data

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Initialize the Replicate model with the API key
client = replicate.Client(replicate_api_token=REPLICATE_API_TOKEN)

# Function to interact with the user and get input
def get_user_input():
    print("Welcome to the AI Pharmacology Assistant.")
    print("Please enter the following patient details.\n")
    medications = input('Enter medications (separated by commas): ')
    side_effects = input('Enter side effects (separated by commas): ')
    medical_condition = input('Enter medical condition: ')
    print("\nThank you. Processing the information...\n")
    return medications, side_effects, medical_condition

# Flag to switch between testing mode and user interaction mode
TESTING_MODE = True  # Set to False for user interaction mode

# Function to process each synthetic case
def process_synthetic_cases(synthetic_cases):
    global test_count
    for case in synthetic_cases:
        test_count += 1  # Increment test count for each case
        input = {
            "prompt": (
                f"As an AI trained in pharmacology, analyze the potential drug-related causes of side effects and the "
                f"impact of the patient's medical condition on their treatment. Medications: {case['medications']}. Reported "
                f"side effects: {case['side_effects']}. Medical condition: {case['medical_condition']}. Provide detailed insights that "
                f"can aid healthcare professionals in making informed treatment decisions."
            ),
            "temperature": 0.2
        }
        # Call the model and get the output
        model_output = client.predict("snowflake/snowflake-arctic-instruct", input)
        # Log the performance
        log_performance(case, model_output, test_count)

# Main application logic
if TESTING_MODE:
    # Generate synthetic data
    synthetic_cases = generate_synthetic_data(num_cases=50)  # Increase the number of cases as needed
    # Process synthetic cases for testing
    process_synthetic_cases(synthetic_cases)
else:
    # Get user input for live interaction
    medications, side_effects, medical_condition = get_user_input()
    # Construct the input for the model
    user_input = {
        "prompt": (
            f"As an AI trained in pharmacology, analyze the potential drug-related causes of side effects and the "
            f"impact of the patient's medical condition on their treatment. Medications: {medications}. Reported "
            f"side effects: {side_effects}. Medical condition: {medical_condition}. Provide detailed insights that "
            f"can aid healthcare professionals in making informed treatment decisions."
        ),
        "temperature": 0.2
    }
    # Call the model and get the output for user input
    user_model_output = client.predict("snowflake/snowflake-arctic-instruct", user_input)
    # Log the performance for user input
    log_performance({'medications': medications, 'side_effects': side_effects, 'medical_condition': medical_condition}, user_model_output, test_count)
    # Display the model output to the user
    print(user_model_output)

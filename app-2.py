import os
import replicate
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging to capture both console output and file output
logging.basicConfig(filename='model_performance.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

test_count = 0  # Initialize test count

def log_performance(input_data, model_output):
    global test_count
    test_count += 1  # Increment test count for each test
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'Test #{test_count} | Timestamp: {timestamp} | Input: {input_data} | Output: {model_output}')

import random
import itertools

# Extended lists of medications, side effects, and medical conditions
medications_list = ['Aspirin', 'Metformin', 'Lisinopril', 'Atorvastatin', 'Simvastatin', 'Levothyroxine']
side_effects_list = ['Nausea', 'Dizziness', 'Headache', 'Fatigue', 'Dry mouth', 'Insomnia']
medical_conditions_list = ['Type 2 Diabetes', 'Hypertension', 'High Cholesterol', 'Hypothyroidism', 'Asthma']

# Function to generate a random combination of medications, side effects, and medical conditions
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

# Generate 10 cases of synthetic data
synthetic_cases = generate_synthetic_data()

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

medications, side_effects, medical_condition = get_user_input()

input = {
    "prompt": (
        f"As an AI trained in pharmacology, analyze the potential drug-related causes of side effects and the "
        f"impact of the patient's medical condition on their treatment. Medications: {medications}. Reported "
        f"side effects: {side_effects}. Medical condition: {medical_condition}. Provide detailed insights that "
        f"can aid healthcare professionals in making informed treatment decisions."
    ),
    "temperature": 0.2
}

for event in replicate.stream(
    "snowflake/snowflake-arctic-instruct",
    input=input
):
    print(event, end="")

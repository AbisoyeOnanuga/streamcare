import replicate
import os
import time
from dotenv import load_dotenv
import logging
from datetime import datetime
import random

# Define loggers at the module level
arctic_logger = logging.getLogger('arctic_instruct')
training_logger = logging.getLogger('training_sim')

def setup_logging():
    # Setup for Arctic Instruct logger
    arctic_logger.setLevel(logging.INFO)
    # Create handlers for the Arctic Instruct logger
    arctic_file_handler = logging.FileHandler('arctic-instruct_performance.log')
    arctic_file_handler.setLevel(logging.INFO)
    # Create formatters and add it to handlers
    arctic_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    arctic_file_handler.setFormatter(arctic_format)
    # Add handlers to the logger
    arctic_logger.addHandler(arctic_file_handler)

    # Setup for Training Simulation logger
    training_logger.setLevel(logging.INFO)
    # Create handlers for the training simulation logger
    training_file_handler = logging.FileHandler('training_sim.log')
    training_file_handler.setLevel(logging.INFO)
    # Create formatters and add it to handlers
    training_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    training_file_handler.setFormatter(training_format)
    # Add handlers to the logger
    training_logger.addHandler(training_file_handler)

# Function to log performance for Arctic Instruct
def log_performance(test_type, model_name, input_data, model_outputs, test_count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"Test Type: {test_type} | Model: {model_name} | Test #{test_count} | Timestamp: {timestamp} | Input: {input_data} | Output: {model_outputs}"
    arctic_logger.info(log_message)

# Function to log performance for training simulation
def training_log_performance(test_type, model_name, input_data, model_outputs, test_count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"Test Type: {test_type} | Model: {model_name} | Test #{test_count} | Timestamp: {timestamp} | Input: {input_data} | Output: {model_outputs}"
    training_logger.info(log_message)
    

# Extended lists for synthetic data generation
medications_list = ['Aspirin', 'Metformin', 'Lisinopril', 'Atorvastatin', 'Simvastatin', 'Levothyroxine']
side_effects_list = ['Nausea', 'Dizziness', 'Headache', 'Fatigue', 'Dry mouth', 'Insomnia']
medical_conditions_list = ['Type 2 Diabetes', 'Hypertension', 'High Cholesterol', 'Hypothyroidism', 'Asthma']

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
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Define your model_name here (e.g., from a config file or another environment variable)
model_name = "snowflake/snowflake-arctic-instruct"

# Function to handle streaming with retries
def stream_with_retries(model_name, input_data, max_retries=3, backoff_factor=1):
    for _ in range(max_retries):
        try:
            # Attempt to stream the response from the model
            for event in client.stream(model_name, input=input_data):
                if hasattr(event, 'data') and event.data.strip().rstrip('{}'):
                    yield event.data  # Yield each part of the model output as it is streamed
            break  # If successful, break out of the retry loop
        except Exception as e:
            logging.exception(f"An error occurred while streaming: {e}")
            print(f"An error occurred: {e}")
            time.sleep(backoff_factor)  # Wait before retrying
            backoff_factor *= 2  # Exponential backoff

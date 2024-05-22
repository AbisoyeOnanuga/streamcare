import os
import replicate
from dotenv import load_dotenv
import logging
from datetime import datetime
import random
import time

# Create a custom logger for the Arctic Instruct performance
arctic_logger = logging.getLogger('arctic_instruct')
arctic_logger.setLevel(logging.INFO)
# Create handlers for the Arctic Instruct logger
arctic_file_handler = logging.FileHandler('arctic-instruct_performance.log')
arctic_file_handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
arctic_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
arctic_file_handler.setFormatter(arctic_format)
# Add handlers to the logger
arctic_logger.addHandler(arctic_file_handler)

# Create a custom logger for the training simulation
training_logger = logging.getLogger('training_sim')
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
    training_logger.info(f"Test Type: {test_type}")
    training_logger.info(f"Model: {model_name}")
    training_logger.info(f"Test Count: {test_count}")
    training_logger.info(f"Input Data: {input_data}")
    training_logger.info(f"Model Outputs: {model_outputs}")

# Initialize test count
test_count = 0

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
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

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
TESTING_MODE = False  # Set to False for user interaction mode
TRAINING_SIMULATION_MODE = False

# Function to process each synthetic case
def process_synthetic_cases(synthetic_cases, model_name, test_type):
    global test_count
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
        # Initialize an empty list to store model outputs for the current case
        model_outputs = []
        relevant_information_generated = False
        
        try:
            # Call the model and collect the output
            for event in client.stream(model_name, input={'prompt': input_prompt, 'temperature': 0.2}):
                if hasattr(event, 'data') and event.data.strip():
                    # Check for and remove any trailing empty dictionary representations
                    cleaned_output = event.data.strip().rstrip('{}')
                    # Append non-empty model output to the list
                    model_outputs.append(cleaned_output)
                    print(cleaned_output)  # Print each part of the model output as it is streamed
                    relevant_information_generated = True
        except Exception as e:
            print(f"An error occurred while streaming: {e}")
            model_outputs.append(f"An error occurred: {e}")

        if not relevant_information_generated:
            # Log only once if no relevant information is generated
            print("No relevant information generated by the model.")
            model_outputs.append("No relevant information generated by the model.")

        # Log the performance for the current case
        log_performance(test_type, model_name, input_data, model_outputs, test_count)

# Call the function with the synthetic cases and the model name
model_name = "snowflake/snowflake-arctic-instruct"  # Replace with the actual model name

# Function to handle streaming with retries
def stream_with_retries(model_name, input_data, max_retries=3, backoff_factor=1):
    for _ in range(max_retries):
        try:
            # Attempt to stream the response from the model
            for event in client.stream(model_name, input=input_data):
                if hasattr(event, 'data') and event.data.strip():
                    yield event.data  # Yield each part of the model output as it is streamed
            break  # If successful, break out of the retry loop
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(backoff_factor)  # Wait before retrying
            backoff_factor *= 2  # Exponential backoff

# Function to interact with the user and collect their diagnosis and treatment plan
def user_diagnosis_and_treatment():
    user_diagnosis = input("Enter your diagnosis: ")
    user_treatment_plan = input("Enter your suggested treatment plan: ")
    return user_diagnosis, user_treatment_plan

# Function to provide AI feedback on the user's input
def ai_feedback_on_user_input(user_diagnosis, user_treatment_plan, scenario):
    # Construct the prompt for AI feedback
    ai_feedback_prompt = (
        f"User Diagnosis: {user_diagnosis}\n"
        f"User Treatment Plan: {user_treatment_plan}\n"
        f"Please provide feedback and additional insights based on the user's input and the following patient information:\n"
        f"Medications: {scenario['medications']}\n"
        f"Reported Side Effects: {scenario['side_effects']}\n"
        f"Medical Condition: {scenario['medical_condition']}\n"
    )

    # Initialize an empty list to store AI feedback
    ai_feedback_outputs = []

    try:
        # Call the model and collect the output
        for event in client.stream(model_name, input={'prompt': ai_feedback_prompt, 'temperature': 0.2}):
            if hasattr(event, 'data') and event.data.strip():
                # Append non-empty model output to the list
                ai_feedback_outputs.append(event.data.strip().rstrip('{}'))
    except Exception as e:
        print(f"An error occurred while streaming: {e}")
        ai_feedback_outputs.append(f"An error occurred: {e}")

    # Combine all parts of the AI feedback into one string
    ai_feedback = ' '.join(ai_feedback_outputs)
    return ai_feedback

# Define the training simulation function
def run_training_simulation(model_name, num_cases=3):
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

# Main application logic
if TESTING_MODE:
    test_type = 'Synthetic'
    synthetic_cases = generate_synthetic_data(num_cases=25)
    process_synthetic_cases(synthetic_cases, model_name, test_type)
elif TRAINING_SIMULATION_MODE:
    test_type = 'Training-simulation'
    run_training_simulation(model_name, num_cases=3)
else:
    test_type = 'User'
    medications, side_effects, medical_condition = get_user_input()
    user_input_prompt = {
        'prompt': (
            f"As an AI trained in pharmacology, analyze the potential drug-related causes of side effects and the "
            f"impact of the patient's medical condition on their treatment. Medications: {medications}. Reported "
            f"side effects: {side_effects}. Medical condition: {medical_condition}. Provide detailed insights that "
            f"can aid healthcare professionals in making informed treatment decisions."
        ),
        'temperature': 0.2
    }
    model_outputs = []
    relevant_information_generated = False

    # Collecting model outputs
    for event in replicate.stream(model_name, input=user_input_prompt):
        if hasattr(event, 'data'):
            model_output = event.data
            if model_output.strip():
                # Check for and remove any trailing empty dictionary representations
                cleaned_output = model_output.strip().rstrip('{}')
                model_outputs.append(cleaned_output)  # Append to the list
                print(cleaned_output, end='')  # Print each part of the model output as it is streamed
                relevant_information_generated = True

    if not relevant_information_generated:
        print("\nNo relevant information generated by the model.")

    # Log the performance after collecting all outputs
    log_performance(test_type, model_name, {'medications': medications, 'side_effects': side_effects, 'medical_condition': medical_condition}, model_outputs, test_count)
    
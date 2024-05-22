import os
import replicate
from dotenv import load_dotenv
import streamlit as st
from streamcare.synthetic_test import run_synthetic_test
from streamcare.training_simulation import run_training_simulation
from streamcare.user_interaction import run_user_interaction


# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Initialize the Replicate model with the API key
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Define your model_name here (e.g., from a config file or another environment variable)
model_name = "snowflake/snowflake-arctic-instruct"

# Initialize the Streamlit app
def main():
    st.title('Medical AI Assistant')

    # Sidebar for mode selection
    mode = st.sidebar.radio("Choose the mode:", ["Synthetic Test", "Training Simulation", "User Interaction"])

    if mode == "Synthetic Test":
        # Synthetic test mode
        st.subheader('Synthetic Test Mode')
        num_cases = st.number_input('Number of synthetic cases:', min_value=1, value=25)
        if st.button('Run Synthetic Test'):
            with st.spinner('Running synthetic tests...'):
                run_synthetic_test(num_cases)
                st.success('Synthetic tests complete!')

    elif mode == "Training Simulation":
        # Training simulation mode
        st.subheader('Training Simulation Mode')
        num_cases = st.number_input('Number of training cases:', min_value=1, value=3)
        if st.button('Run Training Simulation'):
            with st.spinner('Running training simulation...'):
                run_training_simulation(num_cases)
                st.success('Training simulation complete!')

    elif mode == "User Interaction":
        st.subheader('User Interaction Mode')
        medications = st.text_input('Enter medications (separated by commas):')
        side_effects = st.text_input('Enter side effects (separated by commas):')
        medical_condition = st.text_input('Enter medical condition:')
        submit_button = st.button('Submit')

        if submit_button and medications and side_effects and medical_condition:
            with st.spinner('Processing user input...'):
                model_outputs = run_user_interaction(medications, side_effects, medical_condition, model_name)
                for output in model_outputs:
                    st.write(output)
                st.success('User interaction complete!')

# Run the Streamlit app
if __name__ == '__main__':
    main()

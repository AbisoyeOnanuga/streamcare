import streamlit as st
import replicate

# Initialize the Replicate client with your API token
client = replicate.Client(api_token="your_replicate_api_token")

# Streamlit app title
st.title("LLM Pharmacology Analysis App")

# Input fields for user input
medications = st.text_input("Medications", "Aspirin, Metformin")
side_effects = st.text_input("Side Effects", "Nausea, dizziness")
medical_condition = st.text_input("Medical Condition", "Type 2 Diabetes")

# Button to send the input to the API and display the response
if st.button('Analyze'):
    with st.spinner('Analyzing... Please wait.'):
        # Prepare the input for the API call
        input = {
            "prompt": (
                f"As an AI trained in pharmacology, analyze the potential drug-related causes of side effects and the "
                f"impact of the patient's medical condition on their treatment. Medications: {medications}. Reported "
                f"side effects: {side_effects}. Medical condition: {medical_condition}. Provide detailed insights that "
                f"can aid healthcare professionals in making informed treatment decisions."
            ),
            "temperature": 0.2,
            "max_tokens": 150  # Adjust the number of tokens as needed
        }

        # API call to generate the analysis
        response = client.stream("snowflake/snowflake-arctic-instruct", input)
        
        # Display the analysis
        if 'output' in response:
            st.success('Analysis complete:')
            st.write(response['output'])
        else:
            # If 'output' is not in response, it might be an error or status message
            st.error(f"Error or status message received: {response}")

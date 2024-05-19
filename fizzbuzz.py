import os
import replicate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Initialize the Replicate model with the API key
client = replicate.Client(replicate_api_token=REPLICATE_API_TOKEN)

medications = "Aspirin, Metformin"
side_effects = "Nausea, dizziness"
medical_condition = "Type 2 Diabetes"

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
# Streamcare
Streamcare is a cutting-edge application designed to revolutionize primary care by leveraging AI to personalize patient treatment plans.

## Installation

Clone the repository and install the required dependencies:

```
git clone https://github.com/AbisoyeOnanuga/streamcare.git
cd streamcare
pip install -r requirements.txt
pip install replicate  # Install the Replicate package
```

## Usage
Run the app locally with:

```streamlit run main_app.py```

Run the app locally in debug mode with:
```streamlit run main_app.py --logger.level debug```

## Configuration
Set your API keys in the `.env` file or as environment variables:

```REPLICATE_API_TOKEN=your_replicate_api_token```

## AI Model Integration
Streamcare uses the `replicate` package to interact with the `snowflake/snowflake-arctic-instruct` model for generating medical insights.

## Deployment
Deploy Streamcare on Streamlit Cloud by following the instructions in DEPLOYMENT.md.

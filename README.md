# streamcare
Streamcare is a cutting-edge application designed to revolutionize primary care by leveraging AI to personalize patient treatment plans.

## Installation

Clone the repository and install the required dependencies:

```
git clone https://github.com/yourusername/streamcare.git
cd streamcare
pip install -r requirements.txt
pip install replicate  # Install the Replicate package
```

## Usage
Run the app locally with:

```streamlit run app.py```

## Configuration
Set your API keys in the `.env` file or as environment variables:

```GEMINI_KEY=your_gemini_api_key```

## AI Model Integration
Streamcare uses the `replicate` package to interact with the s`nowflake/snowflake-arctic-instruct` model for generating medical insights.

## Deployment
Deploy Streamcare on Streamlit Cloud by following the instructions in DEPLOYMENT.md.
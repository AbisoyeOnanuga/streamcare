# Streamcare

![Streamcare thumbnail](https://github.com/AbisoyeOnanuga/streamcare/assets/102636953/d816558d-4712-4f6c-bb5b-e63b84d73631)

Streamcare is an innovative application that integrates advanced AI to enhance primary care. It provides a platform for healthcare professionals to simulate medical scenarios, analyze medical conditions, and refine treatment plans using AI-generated insights.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [AI Model Integration](#ai-model-integration)

## Installation

To get started with Streamcare, clone the repository and install the necessary dependencies:

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

## Features
- Synthetic Test Mode: Simulate medical cases and receive AI analysis.
- Training Simulation Mode: Engage in medical scenario simulations with AI feedback.
- User Interaction Mode: Interact with AI for medical condition analysis and insights.
- UI/UX Enhancements: Improved user interface for an intuitive experience.

## Directory Structure

- `/streamcare`
    - `carecli.py`:               # Terminal prototype
    - `test.py`:                  # Initial terminal test app
- `main_app.py`:              # Streamlit app
- `training_simulation.py`:   # Training simulation module
- `synthetic_test.py`:        # Synthetic test module
- `user_interaction.py`:      # User interaction module
- `utils.py`:                 # Utility functions
- `Readme.md`:                # Project documentation (current file)
- `requirements.txt`:         # Dependency list

## AI Model Integration
Streamcare uses the `replicate` package to interact with the `snowflake/snowflake-arctic-instruct` model for generating medical insights.

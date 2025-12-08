# Game Dev Assistant

## Overview
The Game Dev Assistant is a tool built to help game developers manage documentation for a project. Users are able to upload game design materials—rules, mechanics descriptions, lore, and other reference documents—and then ask natural-language questions about them. The system retrieves relevant information from the uploaded documents and uses it as context for an LLM. This ensures responses remain grounded in the source material rather than hallucinating.

## How to Run Locally
Note: You may need to use python 3.12 to run this project due to pytorch compatability.
### 1. Clone this repository
```
git clone https://github.com/anthonyhenry/Game-Design-Knowledge-Assistant
cd Game-Design-Knowledge-Assistant
```
### 2. Create a virtual environment in your project directory
```
python -m venv venv
```
### 3. Activate the virtual environment
Windows:
```
venv\Scripts\activate
```
Mac/Linux:
```
source venv/bin/activate
```
### 4. Install dependencies within the virtual environment
```
pip install -r requirements.txt
```
### 5. Create a .env file for your groq api key
This app uses the GROQ API to use the llama-3.1-8b-instant LLM. You will need to supply a GROQ api key. You can acquire one by creating an account at  https://console.groq.com/. Once you have a GROQ API key, you must create a .env file in your directory and supply your API key like so:
```
GROQ_API_KEY = "your_api_key"
```
### 6. Run the app
In your virtual environment run:
```
streamlit run app.py
```

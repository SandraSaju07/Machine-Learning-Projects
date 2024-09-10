# Chatbot Project

## Overview
This project is an interactive chatbot built with Python using Natural Language Processing (NLP) techniques. It leverages the `nltk` library for tokenization and `sklearn` for model training. The chatbot runs on a web interface using Streamlit.

## Project Structure
- `app.py`: The main Streamlit application.
- `train_model.py`: Script to train the chatbot model.
- `data/intents.json`: Holds chatbot intents and responses.
- `model/`: Directory where the trained model and vectorizer are stored.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train_model.py

3. Run the chatbot app:
   streamlit run app.py

## Future Enhancements
- Add more intents and responses.
- Implement real-time data retrieval (e.g., weather, news).
- Deploy the chatbot on a cloud platform.

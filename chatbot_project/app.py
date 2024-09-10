# ChatBot App

import os
import streamlit as st
import random
import pickle
import nltk
import json

# Author: Sandra
# Description: ChatBot Application

nltk.download('punkt')

# Load model and vectorizer
with open(os.path.join('model','chatbot_model.pkl'),'rb') as model_file:
    clf = pickle.load(model_file)
with open(os.path.join('model','vectorizer.pkl'),'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Load intents
with open(os.path.join('data','intents.json'),'rb') as file:
    intents = json.load(file)['intents']

def chatbot_response(input_text):
    input_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_vector)[0]
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

# Streamlit UI
def main():
    st.title('Interactive Chabot')
    st.markdown("### Powered by Natural Language Processing (NLP)")
    st.write("Type a message below to chat with the bot!")

    user_input = st.text_input("You:")

    if user_input:
        response = chatbot_response(user_input)
        st.text_area("Chatbot:",value=response,height=100,max_chars=None)
        if response.lower() in ['bye','goodbye']:
            st.write("Thank you for chatting with me! Have a great day!!")
            st.stop()

if __name__ == "__main__":
    main()

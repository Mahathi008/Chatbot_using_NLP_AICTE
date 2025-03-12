import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issue for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Correct file path to intents.json
file_path = os.path.abspath(r"C:\Users\PINKY\Desktop\Implementation of ChatBot using NLP\intents.json")

# Load intents from JSON file
try:
    with open(file_path, "r", encoding="utf-8") as file:
        intents = json.load(file)  # Load as a list
except FileNotFoundError:
    st.error(f"Error: '{file_path}' not found. Make sure the file exists.")
    intents = []
except json.JSONDecodeError:
    st.error(f"Error: '{file_path}' is not a valid JSON file.")
    intents = []

# Create vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess data
tags = []
patterns = []

if intents:  # Ensure intents list is not empty
    for intent in intents:
        if "patterns" in intent and "tag" in intent:  # Prevent KeyErrors
            for pattern in intent['patterns']:
                tags.append(intent['tag'])
                patterns.append(pattern)

    # Train the model only if there is data
    if patterns:
        x = vectorizer.fit_transform(patterns)
        y = tags
        clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    if not patterns:
        return "Sorry, I am not trained yet."
    
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Sorry, I don't understand."

# Streamlit UI
def main():
    st.title("Chatbot using NLP & Logistic Regression")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! Type a message to chat.")

        # Ensure chat log exists
        log_file = "chat_log.csv"
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("You:", key="user_input")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=100, max_chars=None)

            # Save conversation to log
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.error("No conversation history found.")

    elif choice == "About":
        st.write("""
        This chatbot uses NLP and Logistic Regression to classify user input into predefined intents.
        It is built using **Streamlit** for the UI and trained on a labeled dataset.
        """)

if __name__ == '__main__':
    main()

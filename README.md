# Spam Message Classifier

## Project Overview

This is a **Spam Message Classifier** built using **Naive Bayes** and **Natural Language Processing (NLP)** techniques. The model classifies SMS messages into two categories: **Spam** and **Not Spam (Ham)**. The dataset used for this project contains SMS messages, and the model preprocesses the text by removing punctuation and stopwords to improve classification accuracy.

The classifier uses the **Multinomial Naive Bayes** algorithm for text classification and is built with the Python libraries **scikit-learn**, **pandas**, **numpy**, **streamlit**, and **nltk**.

---

## Features

- **Preprocessing:** Text data is cleaned by removing punctuation and stopwords to ensure better classification performance.
- **Modeling:** Uses **Multinomial Naive Bayes** to classify messages.
- **Streamlit App:** A simple web interface where users can input a message, and the model will predict whether it is **spam** or **ham** (not spam).

---

---

## Installation and Running the App

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/Spam-Message-Classifier.git

2. **Install dependencies:**

Create a virtual environment and install the required packages using the requirements.txt file:

bash
Copy
pip install -r requirements.txt

3. **Run the Streamlit app:**

Navigate to the webapp folder and run the following command to start the Streamlit application:

bash
Copy
streamlit run app.py

# [Sentiment Analysis Tool](https://sentiment-analysis-tool-egldmedkwnuzyzfdnbbewj.streamlit.app/)

## Project Overview
This project is a sentiment analysis tool that classifies text reviews as either **Positive** or **Negative**. The tool uses machine learning techniques, including TF-IDF for text vectorization and a pre-trained model for prediction. The project consists of data ingestion, model training, and a Flask-based web application to serve the model and make predictions.

## Deliverables

### 1. Code Repository

The repository contains the following components:

- **`data_setup.py`**: Script to set up the database and load data.
- **`train_model.py`**: Script that trains the sentiment analysis model.
- **`sapp.py`**: Flask web application that serves the model and allows users to submit reviews for sentiment analysis.
- **`requirements.txt`**: Lists all Python dependencies for the project.

### 2. Database Schema
  
- **Using SQLite**:
  - The database file is named `reviews.db`.
  - Instructions are provided for setting up and populating the database.

---

## Project Setup

### 1. Install Dependencies

To install the necessary dependencies, follow these steps:

1. Clone this repository:

         git clone https://github.com/yourusername/SENTIMENT-ANALYSIS-TOOL.git

2. Navigate into the project directory:

         cd SENTIMENT-ANALYSIS-TOOL

3. Install the Python dependencies using pip:

         pip install -r requirements.txt

This will install all required Python packages listed in the requirements.txt file, including libraries like Flask, scikit-learn, pandas, and others.


## Directory Structure 

      SENTIMENT-ANALYSIS-TOOL/
      │
      ├── app.py                   # Flask app
      ├── train_model.py           # Model training script
      ├── data_setup.py            # Data setup and ingestion script
      ├── requirements.txt         # Python dependencies
      ├── imdb_reviews.db          # SQLite database file (if using SQLite)
      ├── sentiment_model.pkl      # Trained sentiment model
      ├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
      └── README.md                # This file


### Connecting to SQLite3 Database

## Prerequisites
- SQLite3 is built into Python, so you don't need to install anything extra if you're using Python.
- If you need to install SQLite on your machine, you can download it from the [SQLite official website](https://www.sqlite.org/download.html).

## Steps to Connect to SQLite3 Database

### 1. **Verify the Database File**
Ensure that the `.db` file (e.g., `reviews.db`) is present in your project folder.

### 2. **Establish Connection**
In your Python script (e.g., `train_model.py`), import the SQLite3 module.

           import sqlite3
      
          # Step 1: Connect to the SQLite database
          conn = sqlite3.connect('reviews.db')  # Replace with your actual database file name
          cursor = conn.cursor()
          
          # Step 2: Execute a query (example: fetch all records from 'reviews' table)
          cursor.execute("SELECT * FROM reviews;")
          rows = cursor.fetchall()
          
          # Step 3: Process the data
          for row in rows:
              print(row)
          
          # Step 4: Close the connection
          conn.close()

If the .db file is in the same directory as your script, you can simply use 'reviews.db'.
If the .db file is located in a different directory, provide the full path, for example: '/path/to/your/database/reviews.db'.
    

## Data Acquisition
The dataset is loaded from Kaggle. The details are as follows:

[IMDB dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.

### Chosen Model Approach
- For sentiment analysis, we used a machine learning model built with the following approach:

## Preprocessing:
- Text data is cleaned using custom functions to remove unnecessary characters, punctuation, and stop words.
- The cleaned text is then vectorized using **TF-IDF** (Term Frequency-Inverse Document Frequency), which transforms the text data into numerical features suitable for machine learning algorithms.

## Model Type:
- The model is a Logistic Regression classifier, trained on labeled movie reviews data.
- The training data contains reviews labeled as positive or negative based on sentiment.

## Model Training:
- The model is trained on a dataset of movie reviews using scikit-learn. We have used a 80-20 train-test split for model evaluation.

## Model Accuracy:
- After training, the model achieves an accuracy of 88.77% on the test set.

## Prediction:
- The trained model can predict the sentiment of a given review (positive or negative). The prediction is displayed on the web interface after the user submits a review.
  [Check it out here](https://sentiment-analysis-tool-egldmedkwnuzyzfdnbbewj.streamlit.app/)

### Key Results:
- **Accuracy** : 88.77% on the test set.
- **Model Type** : Logistic Regression.

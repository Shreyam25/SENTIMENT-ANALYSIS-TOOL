import streamlit as st
import joblib
from preprocessing import clean_review  # Assuming clean_review is your text cleaning function
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and TF-IDF vectorizer
model = joblib.load('model_acc_0.88776.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')  # Assuming you saved the vectorizer

# Set the title for the Streamlit app
st.title("Sentiment Analysis Tool")

# Create an input text box for the review
review = st.text_area("Enter the review for sentiment analysis:")

# Button to trigger sentiment prediction
if st.button("Analyze Sentiment"):
    if review:
        # Clean the input review
        cleaned_review = clean_review(review)
        
        # Transform the review using the TF-IDF vectorizer
        unseen_data_tfidf = tfidf.transform([cleaned_review])
        
        # Predict sentiment
        prediction = model.predict(unseen_data_tfidf)
        sentiment = "Positive" if prediction[0] == 'positive' else "Negative"
        
        # Display the result
        if sentiment == "Positive":
            st.markdown(f"### Sentiment: **{sentiment}**", unsafe_allow_html=True)
        else:
            st.markdown(f"### Sentiment: **{sentiment}**", unsafe_allow_html=True)
    else:
        st.warning("Please enter a review for analysis.")

from flask import Flask, request, render_template
import joblib
from preprocessing import clean_review  # Import the cleaning function

app = Flask(__name__)

# Load the saved model and pre-fitted TF-IDF vectorizer from file
model = joblib.load('model_acc_0.88776.pkl')  # Load your trained model
tfidf = joblib.load('tfidf_vectorizer.pkl')  # Load the pre-fitted TF-IDF vectorizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    if request.method == 'POST':
        raw_review = request.form['review']
        cleaned_review = clean_review(raw_review)  # Clean the review text

        # Transform the cleaned review using the pre-fitted TF-IDF vectorizer
        unseen_data_tfidf = tfidf.transform([cleaned_review])  # Ensure input is in list format
        prediction = model.predict(unseen_data_tfidf)
        sentiment = "Positive" if prediction[0] == 'positive' else "Negative"
        prediction_text = f"Sentiment: {sentiment}"
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
import pickle
from flask import Flask, request, render_template

# Load your TfidfVectorizer and model
save_cv = pickle.load(open('tfidfvectorizer.pkl', 'rb'))
model = pickle.load(open('Movies Review Classification.pkl', 'rb'))

# Create the Flask application
app = Flask(__name__)

# Define a function to predict sentiment
def predict_sentiment(review):
    # Transform the input review using the loaded TfidfVectorizer
    review_transformed = save_cv.transform([review])
    
    # Predict the sentiment using the loaded model
    prediction = model.predict(review_transformed)
    
    # Determine sentiment and emoji
    sentiment = "Positive" if prediction == 1 else "Negative"
    emoji = "😊" if prediction == 1 else "😔"
    
    return sentiment, emoji

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    review = request.form['review']
    
    # Get sentiment prediction
    sentiment, emoji = predict_sentiment(review)
    
    # Return the result in a rendered HTML template
    return render_template('index.html', review=review, sentiment=sentiment, emoji=emoji)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

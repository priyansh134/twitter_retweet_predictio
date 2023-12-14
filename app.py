import numpy as np
import pandas as pd
from flask import Flask, request, render_template  ,send_file
import pickle
import re
from nltk.corpus import stopwords
app = Flask(__name__, template_folder="templates", static_folder="static")


model = pickle.load(open('model1.pkl', 'rb'))

# Sample function to extract additional features from the tweet
def extract_text_features(tweet):
    # Counting the number of hashtags
    hashtag_count = len(re.findall(r'#\w+', tweet))
    
    # Counting punctuation
    punctuation_count = len(re.findall(r'[!.,;?]', tweet))

    # Counting characters
    char_count = len(tweet)

    # Counting words
    word_count = len(re.findall(r'\b\w+\b', tweet))

    # Counting mentions
    mention_count = len(re.findall(r'@\w+', tweet))

    # Counting acronyms (assuming they are in all caps)
    acronym_count = len(re.findall(r'\b[A-Z]+\b', tweet))

    # Counting URLs
    url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet))

    return hashtag_count, punctuation_count, char_count, word_count, mention_count, acronym_count, url_count

# Sample function to preprocess the tweet
def preprocess_tweet(tweet):
    # Removing stopwords and converting to lowercase
    stop_words = set(stopwords.words('english'))
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = ' '.join([word.lower() for word in tweet.split() if word.lower() not in stop_words])
    return tweet


@app.route('/')
def home():
    return render_template('page.html')
@app.route('/svm')
def svm_page():
    return render_template('svm_page.html')
@app.route('/randomforest')
def randomforest():
    return render_template('randomforest.html')
@app.route('/knn')
def knn():
    return render_template('knn.html')
@app.route('/gb')
def gb():
    return render_template('gb.html')
@app.route('/rnn')
def RNN():
    return render_template('RNN.html')
@app.route('/lstm')
def LSTM():
    return render_template('LSTM.html')
@app.route('/bilstm')
def BILSTM():
    return render_template('BILSTM.html')

@app.route('/dataset/<dataset_name>', methods=['GET'])
def download_dataset(dataset_name):
    if dataset_name == 'Dataset1':
        dataset_path = 'AllTweets16.csv'  # Replace with the actual path to Dataset1
    elif dataset_name == 'Dataset2':
        dataset_path = 'AllTweets16.csv'  # Replace with the actual path to Dataset2
    else:
        return "Invalid dataset name"

    return send_file(dataset_path, as_attachment=True)

@app.route('/predict', methods=['GET'])
def predict():
    tweet = request.args.get('TWEET')

    # Extract additional text features
    hashtag_count, punctuation_count, char_count, word_count, mention_count, acronym_count, url_count = extract_text_features(tweet)

    # Preprocess the tweet
    preprocessed_tweet = preprocess_tweet(tweet)

    # Convert numeric features to numpy array
    numeric_features = np.array([hashtag_count, punctuation_count, char_count, word_count, mention_count, acronym_count, url_count]).reshape(1, -1)

    # Make prediction using the SVM model
    output = model.predict(numeric_features)

    if output == 1:
        out = 'YOUR TWEET WIIL RETWEET'
    else:
        out = 'YOUR TWEET WILL NOT retwet'

    return render_template('out.html', output=out)

if __name__ == "__main__":
    app.run(debug=True)

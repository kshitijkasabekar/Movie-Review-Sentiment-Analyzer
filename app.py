import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model for sentiment analysis
with open('sentiment_analysis_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to preprocess text for sentiment analysis
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to predict sentiment using the loaded model
# Function to predict sentiment using the loaded model
def predict_sentiment(tweet):
    preprocessed_tweet = preprocess_text(tweet)
    vectorized_tweet = vectorizer.transform([preprocessed_tweet])
    prediction = model.predict(vectorized_tweet)
    # Map predictions to positive (1) or negative (0)
    if prediction[0] == 1:
        return 1  # Positive sentiment
    else:
        return 0  # Negative sentiment


# Function to convert sentiment analysis result to DataFrame
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# Function to analyze sentiment of tokens
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result 

# Streamlit UI
st.set_page_config(page_title="NLP Mini Project")

st.title("NLP Mini Project: Movie Reviews Sentiment Analyzer")

menu = ["Home", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("")
    with st.form(key='nlpForm'):
        raw_text = st.text_area("Enter Text Here")
        submit_button = st.form_submit_button(label='Analyze')

    # layout
    col1, col2 = st.columns(2)
    if submit_button:

        with col1:
            st.info("Results")
            # Sentiment Analysis using TextBlob
            sentiment_textblob = TextBlob(raw_text).sentiment
            st.write(sentiment_textblob)

            # Emoji
            if sentiment_textblob.polarity > 0:
                st.markdown("Sentiment: Positive 😊")
            else:
                st.markdown("Sentiment: Negative 😡")
            # else:
            #     st.markdown("Sentiment: Neutral 😐")

            # Dataframe
            result_df = convert_to_df(sentiment_textblob)
            st.dataframe(result_df)

            # Visualization
            c = alt.Chart(result_df).mark_bar().encode(
                x='metric',
                y='value',
                color='metric')
            st.altair_chart(c, use_container_width=True)

        with col2:
            st.info("Token Sentiment")
            # Token level sentiment analysis using VADER
            token_sentiments = analyze_token_sentiment(raw_text)
            st.write(token_sentiments)

else:
    st.subheader("About")
    st.title("Movie Review Sentiment Analyzer")
    tweet = st.text_input("Enter your review:")

    if st.button("Analyze"):
        if tweet:
            sentiment = predict_sentiment(tweet)
            if sentiment == 0:
                st.error("Negative Sentiment")
            else:
                st.success("Positive Sentiment")
        else:
            st.warning("Please enter a review.")

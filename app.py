import streamlit as st
from utils.ai_checker import check_news_authenticity
from utils.ml_predictor import predict_news
from utils.visualizer import generate_charts
import pandas as pd
import requests

# Load models (from models folder)
# The models are saved as .joblib files
import joblib
gbc_model = joblib.load('models/gbc_model.joblib')
dtc_model = joblib.load('models/dtc_model.joblib')
lr_model = joblib.load('models/lr_model.joblib')

# App title and description
st.set_page_config(page_title="News Prediction", page_icon="📊", layout="wide")
st.title("News Prediction")

# Horizontal Slider for switching between sections
section = st.select_slider('Select Section', options=["Latest News", "AI Bot", "News Prediction using ML", "News Analysis"])

if section == "Latest News":
    st.subheader("Latest News")
    
    # Fetch real-time news (use a public API or scraping)
    def get_latest_news():
        url = "https://newsapi.org/v2/top-headlines?country=in&apiKey=YOUR_API_KEY"
        response = requests.get(url)
        articles = response.json()['articles']
        return articles

    news_articles = get_latest_news()
    for article in news_articles:
        st.markdown(f"**{article['title']}**")
        st.write(article['description'])
        st.markdown(f"[Read more]({article['url']})")
        st.write("---")
    
    st.button("Refresh News")

elif section == "AI Bot":
    st.subheader("AI Bot")
    
    user_input = st.text_input("Enter news headline or article:")
    if user_input:
        prediction, related_articles = check_news_authenticity(user_input, gbc_model, dtc_model, lr_model)
        st.write(f"Prediction: **{prediction}**")
        st.write("Related Articles:")
        for article in related_articles:
            st.markdown(f"- [Read more]({article})")
    
elif section == "News Prediction using ML":
    st.subheader("News Prediction using ML")
    
    user_article = st.text_area("Enter a news article:")
    if user_article:
        prediction = predict_news(user_article, gbc_model, dtc_model, lr_model)
        st.write(f"Prediction: **{prediction}**")

elif section == "News Analysis":
    st.subheader("News Analysis")
    
    article_for_analysis = st.text_area("Enter a news article to analyze:")
    if article_for_analysis:
        charts = generate_charts(article_for_analysis)
        st.write(charts)


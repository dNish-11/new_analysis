import streamlit as st
import requests
from utils.ai_checker import check_news_authenticity
from utils.visualizer import generate_charts

# Page config
st.set_page_config(page_title="News Prediction", layout="wide")
st.title("📰 News Prediction App")

# Horizontal slidebar (custom section switcher)
section = st.select_slider(
    "Select Section",
    options=["Latest News", "AI Bot", "News Analysis"]
)

# ---------------- Section 1: Latest News ----------------
if section == "Latest News":
    st.header("🗞️ Latest News")

    def get_latest_news():
        api_key = "AIzaSyAQ0R2le5uzRCjwvOXjr5DQ5AeZdeZNXL4"  # Replace this with your NewsAPI key
        url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={api_key}"
        response = requests.get(url)
        data = response.json()
        return data.get("articles", [])

    articles = get_latest_news()
    for article in articles:
        st.subheader(article["title"])
        st.write(article["description"])
        st.markdown(f"[Read more]({article['url']})")
        st.markdown("---")

    st.button("🔄 Refresh News")

# ---------------- Section 2: AI Bot ----------------
elif section == "AI Bot":
    st.header("🤖 AI Bot – Check News Authenticity")
    
    user_input = st.text_input("Enter news headline or short article:")
    if st.button("Send"):
        if user_input:
            result, links = check_news_authenticity(user_input)
            st.markdown(f"### Result: **{result}**")
            st.write("🔗 Related Articles:")
            for link in links:
                st.markdown(f"- [Read More]({link})")

# ---------------- Section 3: News Analysis ----------------
elif section == "News Analysis":
    st.header("📊 News Analysis – Pictorial View")
    article_text = st.text_area("Paste a news article or content:")
    if st.button("Analyze"):
        if article_text:
            generate_charts(article_text)

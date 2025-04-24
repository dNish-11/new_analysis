import streamlit as st
from streamlit_option_menu import option_menu
import requests
import google.generativeai as genai
import plotly.express as px
import re
import json
from PIL import Image

# --------------- CONFIG ---------------- #
st.set_page_config(page_title="News Prediction", layout="centered")

# Insert your API keys
NEWS_API_KEY = "7d4b56088e5a46868d87c194920750c4"
GEMINI_API_KEY = "AIzaSyAl4rEdDYGSo0DL6Htl2sHmwP3tazBghmc"

# Gemini AI setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------- STYLING ---------------- #
st.markdown("""
    <style>
           
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        text-align: center;
    }
    .block-container {
        padding-top: 2rem;
    }
    .title {
        font-size: 40px;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
        color: #4A90E2;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📰 NEWS PREDICTION</div>', unsafe_allow_html=True)

# --------------- NAVIGATION ---------------- #
selected = option_menu(
    menu_title=None,
    options=["Latest", "Genai Verify", "pic-toricle", "Prediction"],
    icons=["newspaper", "robot", "bar-chart", "magic"],
    orientation="horizontal",
    default_index=0,
)

# --------------- SECTION 1: LATEST NEWS ---------------- #
if selected == "Latest":
    st.subheader("🌐 Latest News")
    topic = st.text_input("Search news by topic (leave blank for Top Headlines)", "")

    if st.button("Fetch News"):
        with st.spinner("Fetching news..."):
            url = f"https://newsapi.org/v2/top-headlines?q={topic}&language=en&apiKey={NEWS_API_KEY}" if topic else f"https://newsapi.org/v2/top-headlines?language=en&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            data = response.json()

            if data["status"] == "ok" and data["totalResults"] > 0:
                for article in data["articles"][:5]:
                    st.markdown(f"#### 🗞️ {article['title']}")
                    st.markdown(f"{article['description']}")
                    st.markdown(f"[Read more]({article['url']})")
                    st.markdown("---")
            else:
                st.error("No news found or API limit reached.")

# --------------- SECTION 2: AI BOT ---------------- #
elif selected == "Genai Verify":
    st.subheader("🤖 Ask Me Anything (powered by Gemini AI)")
    user_question = st.text_input("Type your question")

    if st.button("Get Answer"):
        if user_question.strip():
            with st.spinner("Thinking..."):
                try:
                    # Custom prompt with realism check and summary
                    prompt = f"""Check if the following query is realistic or fake.
Then, generate a short informative summary about it.

Query: "{user_question}"

Respond in two parts:
1. Reality Check: Real or Fake
2. Summary:"""

                    response = model.generate_content(prompt)
                    result = response.text.strip()

                    # Save to history
                    st.session_state.chat_history.append({"user": user_question, "bot": result})

                    st.success("AI Response:")
                    st.write(result)

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question.")

    # Show chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        for chat in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")
            st.markdown("---")

# --------------- SECTION 3: NEWS DASHBOARD ---------------- #
elif selected == "pic-toricle":
    st.subheader("🖼️ News Image Analyzer")
    uploaded_image = st.file_uploader("Upload a news image (JPG)", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image content..."):
                try:
                    prompt = """You are an expert news analyst.
Given this image, analyze it and answer in the following format:

1. Summary of the news (1 paragraph)
2. Detailed information or background
3. Provide 2-3 possible relevant website links

Format your response with headings for each section."""

                    response = model.generate_content([prompt, image])
                    st.markdown("### 🧠 Analysis Result")
                    st.write(response.text)

                except Exception as e:
                    st.error(f"Error analyzing image: {e}")

#------------Section 4: News prediction using ML--------------------

# --------------- SECTION 4: PREDICTION (ML Model) ---------------- #
elif selected == "Prediction":
    import joblib
    import string
    from sklearn.feature_extraction.text import TfidfVectorizer

    st.subheader("🔮 News Category Prediction")
    
    user_input = st.text_area("Enter a news headline or short article:", height=150)

    if st.button("Predict Category"):
        if user_input.strip():
            with st.spinner("Predicting news category..."):
                try:
                    # Sample preprocessing function
                    def preprocess(text):
                        text = text.lower()
                        text = text.translate(str.maketrans('', '', string.punctuation))
                        return text

                    # Load model and vectorizer (assumed pre-trained and saved)
                    model = joblib.load("model/news_category_model.pkl")
                    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

                    # Preprocess and vectorize
                    processed_text = preprocess(user_input)
                    vectorized_input = vectorizer.transform([processed_text])

                    # Make prediction
                    prediction = model.predict(vectorized_input)[0]
                    st.success(f"🗂️ Predicted News Category: **{prediction}**")

                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
        else:
            st.warning("Please enter some text for prediction.")
 
                    

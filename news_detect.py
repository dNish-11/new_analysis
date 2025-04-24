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


# --------------- SECTION 4: PREDICTION (ML Model) ---------------- #
elif selected == "Prediction":
    import joblib
    import string
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd # type: ignore
import re
import streamlit as st # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore

# Load data
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')
true_data['label'] = 1
fake_data['label'] = 0
news = pd.concat([true_data, fake_data], axis=0)
news = news.sample(frac=1).reset_index(drop=True)
news = news.drop(['subject', 'date', 'title'], axis=1)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

news['text'] = news['text'].apply(clean_text)
x = news['text']
y = news['label']

# TF-IDF and model training
vectorizer = TfidfVectorizer(max_features=5000)
xv = vectorizer.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(xv, y, test_size=0.3)

model = LogisticRegression()#1st model
model.fit(x_train, y_train)

model_dtc = DecisionTreeClassifier()#2nd model
model_dtc.fit(x_train, y_train)

# Streamlit App
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter the news article text below:")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]
        prediction_1 = model_dtc.predict(vec_input)[0]

        if prediction == 1 and prediction_1 == 1:
            st.success("✅ It might be TRUE.-----BOTH")
        elif prediction == 1 and prediction_1 == 0:
            st.success("✅ It might be TRUE.----MAIN MODEL")
        elif prediction == 0 and prediction_1 == 1:
            st.success("✅ It might be TRUE.----DTC MODEL")
        else:
            st.error("❌ It may be FAKE NEWS. Please verify it.")
    else:
        st.warning("Please enter some text to analyze.")


 
                    

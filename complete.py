import streamlit as st
from streamlit_option_menu import option_menu
import requests
import google.generativeai as genai
import plotly.express as px
import re
import json
from PIL import Image
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# --------------- CONFIG ---------------- #
st.set_page_config(page_title="PrimeNews ", layout="centered")

import base64

def set_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image path
set_bg_from_local("bg_news_img.jpg")

#----------------------
st.markdown("""
    <style>ok
    body{
       background-image: url('bg_news_img.jpg');
       background-size: cover;        /* Makes the image cover the whole area */
       background-repeat: no-repeat;  /* Prevents the image from repeating */
       background-position: center;   /* Centers the image */
    }
     
    
    /* Background Gradient */
    body {
       background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%) !important;
    }

    /* Container Styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 15px;
    }

    /* Title */
    .title {
        font-size: 40px;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
        color: #1a73e8;
    }

    /* Center text inputs and make them elegant */
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        text-align: center;
        font-size: 16px;
        border-radius: 10px;
        border: 1px solid #ccc;
    }

    /* Option menu styling */
    .css-1d391kg {
        background-color: #f0f4f8 !important;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        font-weight: 500;
    }

    /* Navigation bar hover effects */
    .nav-link {
        color: #333 !important;
        transition: all 0.3s ease-in-out;
    }

    .nav-link:hover {
        color: #1a73e8 !important;
        background-color: #e3ecf8 !important;
        border-radius: 40px;
    }

    .nav-link.active {
        background-color: #d2e3fc !important;
        color: #0b57d0 !important;
        border-radius: 8px;
    }
            
            
    /* Button hover effect */
    .stButton > button {
        border-radius: 10px;
        background-color: #4A90E2;
        color: white;
        transition: background-color 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-color: orange;
        color: white;
        border-radius: 55px;
        transition: background-color 0.3s ease-in-out;
        transition: border-radius 0.7s ease-in-out;    
    }

    </style>
""", unsafe_allow_html=True)

# --------------- SIDE MENU GUIDE ---------------- #
with st.sidebar:
    st.title("📌 PrimeNews corner")
    st.markdown("""
    --------------------------------------------------------
    **1. Latest News**
    - 🔍 Enter a topic in the text field.
    - 📥 Click on "Fetch News".
    - 📰 Top headlines will be displayed.
    ---------------------------------------------------------
    **2. Genai Verify**
    - 🧠 Ask any news-related question.
    - 🤖 Click "Get Answer" to check if it’s real or fake.
    - 💬 View past questions in the chat history.
    ----------------------------------------------------------
    **3. pic-toricle (Image Analyzer)**
    - 🖼️ Upload a JPG or PNG news image.
    - 🔎 Click "Analyze Image".
    - 📋 Get a summary, background, and links.
    ----------------------------------------------------------
    **4. Prediction (Fake News Checker)**
    - ✍️ Paste or type any news text.
    - 🔮 Click "Predict" to verify authenticity.
    - ✅/❌ Results shown from 2 ML models.
    ----------------------------------------------------------
        
    """)

#---------------------

# Insert your API keys
NEWS_API_KEY = "7d4b56088e5a46868d87c194920750c4"
GEMINI_API_KEY = "AIzaSyAl4rEdDYGSo0DL6Htl2sHmwP3tazBghmc"

# Gemini AI setup
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

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

st.markdown('<div class="title">🪼PrimeNews</div>', unsafe_allow_html=True)

# --------------- NAVIGATION ---------------- #
selected = option_menu(
    menu_title=None,
    options=["Latest", "Genai Verify", "pic-toricle", "Prediction"],
    icons=["newspaper", "robot", "bar-chart", "magic"],
    orientation="horizontal",
    default_index=0,
)


#----------------------------------------------------------

# Detect if Connect is triggered from FAB
query_params = st.query_params
if query_params.get("Connect") is not None:
    selected = "Connect"


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

# --------------- SECTION 2: Genai Verify ---------------- #
elif selected == "Genai Verify":
    st.subheader("🤖 Any News or doubt ?")
    user_question = st.text_input("Type your question")

    if st.button("Get Answer"):
        if user_question.strip():
            with st.spinner("Thinking..."):
                try:
                    prompt = f"""Check if the following query is realistic or fake.
Then, generate a short informative summary about it.

Query: "{user_question}"

Respond in two parts:
1. Reality Check: Real or Fake
2. Summary:"""

                    response = gemini_model.generate_content(prompt)
                    result = response.text.strip()

                    st.session_state.chat_history.append({"user": user_question, "bot": result})
                    st.success("AI Response:")
                    st.write(result)

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question.")

    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        for chat in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")
            st.markdown("---")

# --------------- SECTION 3: NEWS IMAGE ANALYZER ---------------- #
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

                    response = gemini_model.generate_content([prompt, image])
                    st.markdown("### 🔍 Analysis Result")
                    st.write(response.text)

                except Exception as e:
                    st.error(f"Error analyzing image: {e}")

# --------------- SECTION 4: FAKE NEWS PREDICTION ---------------- #
elif selected == "Prediction":
    st.subheader("🔮 News Prediction")

    @st.cache_resource
    def load_data_and_models():
        true_data = pd.read_csv('D:\minipro\sem-VIth pro\data\True.csv')
        fake_data = pd.read_csv('D:\minipro\sem-VIth pro\data\Fake.csv')
        true_data['label'] = 1
        fake_data['label'] = 0
        news = pd.concat([true_data, fake_data], axis=0)
        news = news.sample(frac=1).reset_index(drop=True)
        news = news.drop(['subject', 'date', 'title'], axis=1)

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

        vectorizer = TfidfVectorizer(max_features=5000)
        xv = vectorizer.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(xv, y, test_size=0.3)

        model_lr = LogisticRegression()
        model_lr.fit(x_train, y_train)

        model_dtc = DecisionTreeClassifier()
        model_dtc.fit(x_train, y_train)

        return vectorizer, model_lr, model_dtc, clean_text

    vectorizer, model_lr, model_dtc, clean_text = load_data_and_models()

    user_input = st.text_area("Enter the news article text below:")

    if st.button("Predict"):
        if user_input:
            cleaned = clean_text(user_input)
            vec_input = vectorizer.transform([cleaned])
            prediction_lr = model_lr.predict(vec_input)[0]
            prediction_dtc = model_dtc.predict(vec_input)[0]

            if prediction_lr == 1 and prediction_dtc == 1:
                st.success("✅ It might be TRUE.          BOTH model predict it's TRUE")
            elif prediction_lr == 1 and prediction_dtc == 0:
                st.success("✅ It might be TRUE.          LR_MODEL predict it's TRUE")
            elif prediction_lr == 0 and prediction_dtc == 1:
                st.success("✅ It might be TRUE.          DTC_MODEL predict it's TRUE")
            else:
                st.error("❌ It might be FAKE. Please verify it.")
        else:
            st.warning("Please enter some text to analyze.")

#-----------------------------Footer------------------------------------------
st.markdown("""

    <style>
            
    .footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 2em;
    }        
    </style>

""", unsafe_allow_html=True)
 
#------------------------------------------------------------------------------

st.markdown('<div class="footer">© 2025 PrimeNews • Built by PrimeDev with ❤️ using Streamlit</div>', unsafe_allow_html=True)








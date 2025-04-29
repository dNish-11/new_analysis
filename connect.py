import streamlit as st

st.set_page_config(page_title="Connect - PrimeNews", layout="centered")

# --------- CSS Styling --------- #
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: #1a73e8;
        margin-bottom: 0.5em;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        margin-bottom: 2em;
        color: #555;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        border: 1px solid #ccc;
    }

    .stButton > button {
        border-radius: 8px;
        background-color: #1a73e8;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #155ab6;
        color: #fff;
    }

    .footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 2em;
    }
    </style>
""", unsafe_allow_html=True)

# --------- Page Title --------- #
st.markdown('<div class="title">📬 Connect with PrimeNews</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">We value your feedback and questions!</div>', unsafe_allow_html=True)

# --------- Contact Form --------- #
st.header("📞 Contact Us")

name = st.text_input("Your Name")
email = st.text_input("Your Email")
message = st.text_area("Your Message")

if st.button("Send Message"):
    if name and email and message:
        st.success("✅ Message sent! We’ll get back to you soon.")
    else:
        st.warning("Please fill out all fields.")

# --------- Feedback Form --------- #
st.header("📝 Feedback")

rating = st.radio("How do you rate PrimeNews?", ["⭐ Poor", "⭐⭐ Fair", "⭐⭐⭐ Good", "⭐⭐⭐⭐ Very Good", "⭐⭐⭐⭐⭐ Excellent"])
suggestions = st.text_area("Any Suggestions?")

if st.button("Submit Feedback"):
    st.success("✅ Thank you for your feedback!")

# --------- Footer --------- #
st.markdown('<div class="footer">© 2025 PrimeNews • Built with ❤️ using Streamlit</div>', unsafe_allow_html=True)

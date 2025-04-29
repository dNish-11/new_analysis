import streamlit as st

# Simulated user database (use Firebase or SQLite in real projects)
users = {"admin": "admin123"}  # username: password

def login_signup_page():
    st.title("🔐 Welcome to PrimeNews")
    choice = st.selectbox("Login or Sign Up", ["Login", "Sign Up"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Sign Up":
        if st.button("Create Account"):
            if username in users:
                st.warning("🚫 Username already exists.")
            else:
                users[username] = password
                st.success("✅ Account created. Please login now.")
    else:
        if st.button("Login"):
            if users.get(username) == password:
                st.success("✅ Login successful.")
                st.session_state.logged_in = True
                st.session_state.username = username
            else:
                st.error("🚫 Invalid username or password.")

    return st.session_state.get("logged_in", False)

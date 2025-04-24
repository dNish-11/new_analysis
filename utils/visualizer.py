import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def generate_charts(text):
    # Dummy word frequency visualization
    words = text.lower().split()
    word_freq = pd.Series(words).value_counts().head(5)

    st.subheader("📈 Top 5 Words in Article")
    fig, ax = plt.subplots()
    word_freq.plot(kind='barh', ax=ax, color='skyblue')
    ax.invert_yaxis()
    st.pyplot(fig)

    # Dummy sentiment chart
    st.subheader("📊 Sentiment Distribution")
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [40, 30, 30]
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

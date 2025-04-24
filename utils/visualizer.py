import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_charts(news_article):
    # Analyze the news article (could be sentiment analysis, keyword frequency, etc.)
    # Here we'll just generate dummy charts
    
    # Sample data for charts
    data = {
        'sentiment': ['Positive', 'Negative', 'Neutral'],
        'percentage': [45, 35, 20]
    }
    
    df = pd.DataFrame(data)
    
    # Create pie chart
    fig, ax = plt.subplots()
    ax.pie(df['percentage'], labels=df['sentiment'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    plt.title("Sentiment Distribution")
    
    # Show the chart
    st.pyplot(fig)
    return fig

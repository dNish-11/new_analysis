import random

def check_news_authenticity(text):
    # Dummy AI-based response (you can connect it to Gemini/ChatGPT API later)
    prediction = random.choice(["Real", "Fake"])
    
    # Dummy related articles
    links = [
        "https://www.thehindu.com/",
        "https://www.indiatoday.in/",
        "https://www.ndtv.com/"
    ]
    return prediction, links

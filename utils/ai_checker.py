import requests

def check_news_authenticity(news_input, gbc_model, dtc_model, lr_model):
    # Check using AI model (for simplicity, use all three models)
    # Return prediction result (Fake or True)
    
    # Example: You can combine the prediction from multiple models to make a final decision
    prediction_gbc = gbc_model.predict([news_input])
    prediction_dtc = dtc_model.predict([news_input])
    prediction_lr = lr_model.predict([news_input])

    # Majority voting logic or any other combination approach
    predictions = [prediction_gbc, prediction_dtc, prediction_lr]
    final_prediction = "Fake" if predictions.count("Fake") > 1 else "True"
    
    # Fetch related articles via an API (for simplicity using a dummy API)
    related_articles = fetch_related_articles(news_input)
    
    return final_prediction, related_articles

def fetch_related_articles(query):
    # Here you can use a news API to fetch related articles
    # Using a dummy URL and returning sample data
    related_articles = [
        "https://www.example.com/article1",
        "https://www.example.com/article2"
    ]
    return related_articles

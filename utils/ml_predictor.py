def predict_news(news_input, gbc_model, dtc_model, lr_model):
    # Use the model to predict news authenticity (True or Fake)
    prediction_gbc = gbc_model.predict([news_input])
    prediction_dtc = dtc_model.predict([news_input])
    prediction_lr = lr_model.predict([news_input])
    
    # Return prediction (you can combine the predictions from models)
    if prediction_gbc == "Fake" and prediction_dtc == "Fake" and prediction_lr == "Fake":
        return "Fake"
    else:
        return "True"

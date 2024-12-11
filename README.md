# Serenity-C242-PS479-ML

## Deep Learning Project with TensorFlow 💻🧠
This repository contains Jupyter notebooks for two key features:

1. **Break Reminders ⏰** – Generates datasets using code to remind users to take breaks after extended screen usage.
2. **Sentiment Analysis 💬** – Analyzes and classifies the sentiment of text data by reading a CSV file uploaded to Google Colab.

## Usage
Break Reminders: Run the break_reminders.ipynb notebook to generate datasets and set up break reminders. Below is the code to predict whether a user needs a break reminder:

```bash
def predict_new_data(input_data):
    new_data_scaled = scaler.transform(input_data)
    prediction = loaded_model.predict(new_data_scaled)
    prediction = (prediction > 0.5).astype(int)  # Threshold, adjust as needed
    return prediction[0][0]

input_data = pd.DataFrame({
    'Age': [25],  # Example age
    # Social media usage data
    'Facebook': [1],
    'Instagram':[0],
    'Reddit':   [0],
    'Threads':  [0],
    'TikTok':   [0],
    'X':        [0],
    'YouTube':  [0],
    '00:00:00': [1],
    '01:00:00': [1],
    '02:00:00': [0],
    '03:00:00': [0],
    '04:00:00': [0],
    '05:00:00': [0],
    '06:00:00': [1],
    '07:00:00': [1],
    '08:00:00': [1],
    '09:00:00': [1],
    '10:00:00': [0],
    '11:00:00': [1],
    '12:00:00': [1],
    '13:00:00': [0],
    '14:00:00': [0],
    '15:00:00': [0],
    '16:00:00': [0],
    '17:00:00': [0],
    '18:00:00': [0],
    '19:00:00': [0],
    '20:00:00': [0],
    '21:00:00': [0],
    '22:00:00': [0],
    '23:00:00': [0]
})
predicted_nn = predict_new_data(input_data)
print(f"Pengguna {'Need Break Reminder!' if predicted_nn == 1 else 'No Need Break Reminder.'}")
```

Sentiment Analysis: Run the sentiment_analysis.ipynb notebook. Upload your CSV file to Google Colab, and it will analyze the sentiment of the text data. Below is the code to predict sentiment with probability:
```bash
def predict_sentiment_with_proba(text, model, tokenizer, max_seq_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)
    sentiment_map = {0: "Positive 😊", 1: "Neutral 😐", 2: "Negative 😞"}
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return sentiment_map[predicted_class], confidence
text = "cantik banget sayang"
sentiment, confidence = predict_sentiment_with_proba(text, model, tokenizer, max_seq_length)
print(f"Prediksi Sentimen: {sentiment}, Confidence: {confidence:.2f}")
```


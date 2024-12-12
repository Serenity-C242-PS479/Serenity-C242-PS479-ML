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

## Deployment
The created model is deployed using Google Cloud Functions. Below are the steps to deploy the model:

### 1. Prepare Your Environment
Ensure that you have:
- A Google Cloud Platform (GCP) account.
- The Google Cloud SDK installed on your local machine.
- Access to a GCP project where you have permission to deploy cloud functions.
- A bucket in Google Cloud Storage containing the model and scaler files.

### 2. Create the Cloud Function Code
Save the provided Python code in a file, e.g., `main.py`. This file contains the logic for downloading the model and scaler, processing input data, and making predictions.

### 3. Prepare the Requirements File
Create a `requirements.txt` file with the necessary dependencies. Example:
```plaintext
google-cloud-storage==2.14.0
tensorflow==2.17.1
pandas==2.1.1
scikit-learn==1.3.1
```
This file ensures that the required libraries are installed in the Cloud Function environment.

### 4. Deploy the Cloud Function
Use the following steps to deploy the function:

#### 1. Navigate to the Directory
Go to the directory where main.py and requirements.txt are stored:
```bash
cd /path/to/your/code
```

#### 2. Deploy the Cloud Function
Use the gcloud CLI to deploy the function:
```bash
gcloud functions deploy predictSocialMediaUsage \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --region asia-southeast-2 \
    --entry-point predict_social_media_usage
```
- --runtime: Specifies the runtime (e.g., Python 3.11).
- --trigger-http: Indicates that the function will be triggered via HTTP.
- --allow-unauthenticated: Makes the function publicly accessible.
- --region: Sets the deployment region (e.g., us-central1).
- --entry-point: Points to the main function (in this case, predict_social_media_usage).

### 5. Test the Cloud Function
Once deployed, the function will return a URL. You can test it using tools like curl or Postman. Example Request:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "Age": 25,
        "Facebook": 1,
        "Instagram": 0,
        "Reddit": 1,
        "Threads": 0,
        "TikTok": 1,
        "X": 0,
        "YouTube": 1,
        "00:00:00": 0.5,
        "01:00:00": 0.2,
        ...
        "23:00:00": 0.1
      }' \
  https://<your-cloud-function-url>
```
Example Response:
```json
{
  "status": "istirahat!"
}
```
## API URL
### 1. API Break Reminder URL:
```bash
https://asia-southeast2-c242-ps479.cloudfunctions.net/serenity-break-reminder-function
```
Request Payload:
```json
{
    "Age": 25,
    "Facebook": 1,
    "Instagram": 0,
    "Reddit": 1,
    "Threads": 1,
    "TikTok": 0,
    "X": 0,
    "YouTube": 0,
    "00:00:00": 1,
    "01:00:00": 0,
    "02:00:00": 0,
    "03:00:00": 0,
    "04:00:00": 0,
    "05:00:00": 0,
    "06:00:00": 1,
    "07:00:00": 1,
    "08:00:00": 0,
    "09:00:00": 1,
    "10:00:00": 0,
    "11:00:00": 1,
    "12:00:00": 1,
    "13:00:00": 0,
    "14:00:00": 0,
    "15:00:00": 0,
    "16:00:00": 0,
    "17:00:00": 0,
    "18:00:00": 0,
    "19:00:00": 0,
    "20:00:00": 0,
    "21:00:00": 0,
    "22:00:00": 0,
    "23:00:00": 0
}
```
Response:
```json
{
    "status": "tidak perlu istirahat"
}
```
### 2. API Predict Sentiment URL:
```bash
https://asia-southeast2-c242-ps479.cloudfunctions.net/serenity-sentiment-function
```
Request Payload:
```json
{
    "text": "Kamu lagi apa ?"
}
```
Response:
```json
{
    "sentiment": "Positive"
}
```

import requests

url = "http://127.0.0.1:8000/predict_mood"

message = {
    "text": "I am so happy today!"
}

response = requests.post(url, json=message)

if response.status_code == 200:
    result = response.json()
    print("Predicted Mood:", result["predicted_mood"])
    print("Confidence Score:", result["confidence_score"])
else:
    print("Error:", response.status_code, response.text)

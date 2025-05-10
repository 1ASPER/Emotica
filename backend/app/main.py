from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

class Message(BaseModel):
    text: str

model_path = "C:/Users/bapanov_n.AU.000/Desktop/Emotica/backend/emotion_model"
emotion_classifier = pipeline("text-classification", model=model_path)

@app.post("/predict_mood")
def predict_mood(message: Message):

    predictions = emotion_classifier(message.text)
    
    predicted_label = predictions[0]['label']
    predicted_score = predictions[0]['score']
    
    return {"message": message.text, "predicted_mood": predicted_label, "confidence_score": predicted_score}
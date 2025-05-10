from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

model_path = "C:/Users/bapanov_n.AU.000/Desktop/Emotica/backend/emotion_model"
emotion_classifier = pipeline("text-classification", model=model_path)

LABEL_TO_EMOTION = {
    "LABEL_0": "Sadness",   # sadness (0)
    "LABEL_1": "Joy",       # joy (1)
    "LABEL_2": "Love",      # love (2)
    "LABEL_3": "Anger",     # anger (3)
    "LABEL_4": "Fear",      # fear (4)
    "LABEL_5": "Surprise"   # surprise (5)
}


@app.post("/predict_mood")
async def predict_mood(message: dict):
    text = message.get("text")
    prediction = emotion_classifier(text)
    predicted_label = prediction[0]['label']
    predicted_mood = LABEL_TO_EMOTION.get(predicted_label, "Unknown")
    confidence_score = prediction[0]['score']
    
    return {
        "predicted_mood": predicted_mood,
        "confidence_score": confidence_score
    }
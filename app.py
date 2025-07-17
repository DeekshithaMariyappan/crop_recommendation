
import gradio as gr
import pickle
import numpy as np
import pandas as pd

# Load model, scaler, and label encoder
with open("crop_recommendation.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Define prediction function
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    crop_name = le.inverse_transform(prediction)[0]
    return f"🌱 Recommended Crop: {crop_name}"

# Gradio interface
interface = gr.Interface(
    fn=predict_crop,
    inputs=[
        gr.Number(label="Nitrogen (N)"),
        gr.Number(label="Phosphorus (P)"),
        gr.Number(label="Potassium (K)"),
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Humidity (%)"),
        gr.Number(label="pH"),
        gr.Number(label="Rainfall (mm)")
    ],
    outputs="text",
    title="🌾 Crop Recommendation System",
    description="Enter soil and climate values to get the most suitable crop."
)

interface.launch()

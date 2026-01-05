import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
from keras_image_helper import create_preprocessor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smoke and Fire Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://smoke-fire-classifier.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model
session = ort.InferenceSession(
    "smoke_fire_classifier.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASSES = ['Smoke', 'fire', 'non fire']


def preprocess_pytorch(X):
    X = np.expand_dims(X, axis=0)

    X = X / 255.0

    X = X.transpose(0, 3, 1, 2)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    mean_tensor = mean.reshape(1, 3, 1, 1)
    std_tensor = std.reshape(1, 3, 1, 1)

    X = (X - mean_tensor) / std_tensor

    return X.astype(np.float32)


preprocessor = create_preprocessor(preprocess_pytorch, target_size=(224, 224))


@app.get("/")
async def root():
    return {"status": "ok", "message": "Smoke and Fire Classifier is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # x = preprocess_image(img)
    img = img.resize((224, 224))
    x = preprocessor.preprocess(img)

    outputs = session.run([output_name], {input_name: x})
    logits = outputs[0][0]

    # Softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    class_id = int(np.argmax(probs))

    return {
        "predicted_class": CLASSES[class_id],
        "confidence": float(probs[class_id])
    }

from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- CONFIG ----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 128

# ---- TRANSFORM ----
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ---- LOAD MODEL ----
model = timm.create_model('resnet18', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---- FASTAPI APP ----
app = FastAPI()

classes = ["Real", "Fake"]

@app.get("/")
def home():
    return {"message": "Fake Image Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(img)

    img = val_transform(image=img)["image"]
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return {
        "prediction": classes[pred],
        "confidence": float(probs[0][pred])
    }
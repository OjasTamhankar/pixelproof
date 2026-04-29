from flask import Flask, jsonify, render_template, request
import torch
import timm
import numpy as np
from PIL import Image, UnidentifiedImageError
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

DEVICE = 'cpu'

# ---- LOAD MODEL ----
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("model_efficientnet_b0.pth", map_location=DEVICE))
model.eval()

# ---- TRANSFORM ----
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ---- ROUTES ----
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist("images")
    wants_json = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    )

    if not files or files[0].filename == '':
        error_message = "No images uploaded"
        if wants_json:
            return jsonify({"error": error_message, "results": []}), 400
        return render_template('index.html', error=error_message)

    results = []
    valid_count = 0

    for index, file in enumerate(files):
        filename = file.filename or f"image-{index + 1}"
        try:
            img = Image.open(file).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            results.append({
                "index": index,
                "filename": filename,
                "error": "Invalid image file"
            })
            continue

        img_np = np.array(img)
        img_tensor = transform(image=img_np)["image"].unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, 1).item()

        classes = ["Real", "Fake"]
        valid_count += 1

        results.append({
            "index": index,
            "filename": filename,
            "prediction": classes[pred].upper(),
            "confidence": round(float(probs[0][pred]) * 100, 2),
            "path": filename
        })

    if valid_count == 0:
        error_message = "No valid image files were uploaded"
        if wants_json:
            return jsonify({"error": error_message, "results": results}), 400
        return render_template('index.html', error=error_message, results=results)

    if wants_json:
        return jsonify({"results": results})

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

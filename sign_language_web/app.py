import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from rembg import remove

import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
import json
from torchvision.models import resnet18
import numpy as np

# ====== Model định nghĩa (nếu cần CNN riêng) ======
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128), torch.nn.ReLU(), torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128*8*8, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ====== Config ======
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'sign_language_model_cnn.pth'
CLASS_JSON = 'class_names.json'
IMG_SIZE = 64

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ====== Load class names ======
with open(CLASS_JSON, 'r', encoding='utf-8') as f:
    class_names = json.load(f)

# ====== Hand detector + crop ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def crop_hand(image_bgr):
    try:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert("RGBA")
        img_no_bg = remove(pil_img)

        arr = np.array(img_no_bg)
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        pad = 20
        x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, arr.shape[1])
        y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, arr.shape[0])
        cropped = img_no_bg.crop((x1, y1, x2, y2))

        # Dán lên nền trắng để xử lý ảnh
        bg = Image.new("RGB", cropped.size, (255, 255, 255))
        bg.paste(cropped, mask=cropped.split()[3])
        return bg

    except Exception as e:
        print(f"[ERROR - crop_hand]: {e}")
        return None

# ====== Transform & Model ======
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = resnet18(pretrained=False)
# model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model= SimpleCNN(num_classes=len(class_names))

# Load model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ====== Routes ======
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        img_bgr = cv2.imread(path)
        hand_pil = crop_hand(img_bgr)
        if hand_pil is None:
            prediction = "No hand detected"
        else:
            tensor = transform(hand_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.nn.functional.softmax(out, dim=1)
                top3_prob, top3_idx = torch.topk(probs, 3)
                prediction = [(class_names[i], float(p)*100) for i, p in zip(top3_idx[0], top3_prob[0])]

            hand_pil.save(path)

    return render_template('index.html', prediction=prediction, filename=filename)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    raw_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(raw_path)

    img_bgr = cv2.imread(raw_path)
    hand_pil = crop_hand(img_bgr)

    if hand_pil is None:
        return jsonify(prediction=None, image_url=None)

    tensor = transform(hand_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)
        top3_prob, top3_idx = torch.topk(probs, 3)
        results = [(class_names[i], float(p)*100) for i, p in zip(top3_idx[0], top3_prob[0])]

    # Save processed image
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    hand_pil.save(output_path)

    return jsonify(
        prediction=results,
        image_url=url_for('static', filename='uploads/' + output_filename)
    )

if __name__ == '__main__':
    app.run(debug=True)
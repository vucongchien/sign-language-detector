import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
import json
from torchvision.models import resnet18

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
MODEL_PATH = 'sign_language_model_resnes.pth'
CLASS_JSON = 'class_names.json'
IMG_SIZE = 64

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ====== Flask ======
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ====== Load class names ======
with open(CLASS_JSON, 'r', encoding='utf-8') as f:
    class_names = json.load(f)

# ====== Mediapipe hand detector ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def crop_hand(image_bgr):
    """Phát hiện và crop tay, trả về PIL Image."""
    h, w, _ = image_bgr.shape
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    # tìm bounding box
    x_min, y_min = w, h
    x_max = y_max = 0
    for lm in results.multi_hand_landmarks[0].landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)
    pad = 200
    x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
    x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)
    crop = image_bgr[y_min:y_max, x_min:x_max]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# ====== Transforms & Model load ======
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 26)
# model = SimpleCNN(len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ====== Routes ======
@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        fname = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(path)

        # read & crop
        img_bgr = cv2.imread(path)
        hand_pil = crop_hand(img_bgr)
        if hand_pil is None:
            prediction = "No hand detected"
        else:
            # transform + predict
            tensor = transform(hand_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.nn.functional.softmax(out, dim=1)
                top3_prob, top3_idx = torch.topk(probs, 3)
                prediction = [(class_names[i], float(p)*100) for i, p in zip(top3_idx[0], top3_prob[0])]


            # lưu lại ảnh đã crop để show
            hand_pil.save(path)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

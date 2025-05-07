import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import json  
from model import SimpleCNN 
from torchvision.models import resnet18

# === 0. Cấu hình chung ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === 1. Load class_names ===
with open('sign_language_web/class_names.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
num_classes = len(class_names)


# === 2. Load mô hình đã huấn luyện ===
# model = SimpleCNN(num_classes).to(device)
# Nếu muốn dùng ResNet18 pretrained, bỏ comment khối này và comment CNN trên:
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('sign_language_web/sign_language_model_resnes.pth', map_location=device))
model.eval()

# === 3. Transform giống lúc validation ===
eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])

# === 4. Mediapipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === 5. Cắt tay và áp nền trắng ===
def segment_hand(image_bgr):
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    lm = results.multi_hand_landmarks[0].landmark
    pts = np.array([(int(p.x * w), int(p.y * h)) for p in lm], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.dilate(mask, None, iterations=5)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    fg = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    bg = np.ones_like(image_bgr, dtype=np.uint8) * 255
    out = np.where(mask[:, :, None] == 0, bg, fg)
    return out

# === 6. TTA: dự đoán với 2 biến thể ảnh ===
tta_transforms = [
    eval_transform,
    transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
]

def predict_tta(model, img_bgr):
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    votes = torch.zeros(num_classes, device=device)
    for t in tta_transforms:
        x = t(img_pil).unsqueeze(0).to(device)
        out = model(x)
        votes += torch.softmax(out, dim=1).squeeze()

    prob = votes / votes.sum()  # Normalize
    top3 = torch.topk(prob, k=3)  # lấy 3 giá trị lớn nhất
    top3_indices = top3.indices.tolist()
    top3_probs = (top3.values * 100).tolist()
    
    return [(class_names[i], p) for i, p in zip(top3_indices, top3_probs)]

# === 7. Load class_names ===
with open('sign_language_web/class_names.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)

# === 8. Mở webcam và xử lý real-time ===
cap = cv2.VideoCapture(0)
print("Press 'c' to capture & predict, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Webcam - press c to capture', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        hand_img = segment_hand(frame)
        if hand_img is None:
            print("No hand detected, please try again!")
        else:
            cv2.imshow('Segmented Hand', hand_img)
            top3_preds = predict_tta(model, hand_img)
            print("Top-3 Predictions:")
            for label, conf in top3_preds:
                print(f" - {label}: {conf:.2f}%")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

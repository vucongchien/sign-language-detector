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
from model import SimpleCNN

# === 0. Cấu hình chung ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === 1. Load class_names ===
with open('sign_language_web/class_names.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
num_classes = len(class_names)

# === 2. Load mô hình đã huấn luyện ===
model = SimpleCNN(num_classes).to(device)
# Nếu muốn dùng ResNet18 pretrained, bỏ comment khối này và comment CNN trên:
# model = resnet18(pretrained=False)
# model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('sign_language_web/sign_language_model_cnn.pth', map_location=device))
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
        return image_bgr  # Hiện khung hình gốc thay vì trắng hoàn toàn

    # Tìm bounding box quanh tay
    lm = results.multi_hand_landmarks[0].landmark
    x_coords = [int(p.x * w) for p in lm]
    y_coords = [int(p.y * h) for p in lm]

    x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
    y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

    # Tạo ảnh trắng
    white_bg = np.ones_like(image_bgr) * 255

    # Copy vùng bàn tay từ ảnh gốc vào nền trắng
    white_bg[y_min:y_max, x_min:x_max] = image_bgr[y_min:y_max, x_min:x_max]
    
    return white_bg


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

# === 7. Mở webcam và xử lý real-time ===
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý cắt tay từ mỗi khung hình
    hand_img = segment_hand(frame)
    
    # Nếu có tay được phát hiện, thực hiện dự đoán
    if hand_img is not None:
        top3_preds = predict_tta(model, hand_img)
        print("Top-3 Predictions:")
        for label, conf in top3_preds:
            print(f" - {label}: {conf:.2f}%")
        
        # Hiển thị kết quả dự đoán lên khung hình
        result_text = "\n".join([f"{label}: {conf:.2f}%" for label, conf in top3_preds])
        cv2.putText(hand_img, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Hiển thị khung hình đã được xử lý
        cv2.imshow('Segmented Hand', hand_img)
    else:
        # Nếu không phát hiện tay, hiển thị khung hình gốc
        cv2.imshow('Webcam - Hand not detected', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
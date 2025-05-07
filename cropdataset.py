import os
from rembg import remove
from PIL import Image
import numpy as np

# --- Cấu hình ---
INPUT_DIR = 'dataset_sieucappro'             # thư mục gốc chứa train/ test
OUTPUT_DIR = 'dataset_sieucappro_rembg'      # nơi lưu ảnh đã xử lý
SPLITS = ['train', 'test']                   # các split
PADDING = 20                                  # thêm viền xung quanh bbox

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in SPLITS:
    in_split = os.path.join(INPUT_DIR, split)
    out_split = os.path.join(OUTPUT_DIR, split)
    os.makedirs(out_split, exist_ok=True)

    for cls in os.listdir(in_split):
        in_cls = os.path.join(in_split, cls)
        out_cls = os.path.join(out_split, cls)
        os.makedirs(out_cls, exist_ok=True)
        
        for fname in os.listdir(in_cls):
            in_path = os.path.join(in_cls, fname)
            out_path = os.path.join(out_cls, os.path.splitext(fname)[0] + '.png')
            try:
                # 1. Load ảnh và remove background
                img = Image.open(in_path).convert("RGBA")
                print(f"Processing {in_path} -> {out_path}")
                img_no_bg = remove(img)  # PIL Image, mode RGBA

                # 2. Tính bounding box từ alpha channel
                arr = np.array(img_no_bg)
                alpha = arr[:, :, 3]
                ys, xs = np.where(alpha > 0)
                if len(xs) == 0 or len(ys) == 0:
                    continue  # nếu không có vùng nào, bỏ qua

                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                # thêm padding
                x1 = max(x1 - PADDING, 0)
                y1 = max(y1 - PADDING, 0)
                x2 = min(x2 + PADDING, arr.shape[1])
                y2 = min(y2 + PADDING, arr.shape[0])

                # 3. Crop vùng tay
                cropped = img_no_bg.crop((x1, y1, x2, y2))

                # 4. (Tuỳ chọn) Chuyển về RGB với nền trắng
                bg = Image.new("RGB", cropped.size, (255, 255, 255))
                bg.paste(cropped, mask=cropped.split()[3])  # dùng alpha làm mask

                # 5. Lưu ảnh PNG (nền trắng) hoặc PNG có alpha nếu bạn muốn
                bg.save(out_path)  

            except Exception as e:
                print(f"Error processing {in_path}: {e}")

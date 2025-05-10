import os
from rembg import remove
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import chain
from pathlib import Path

# --- Cấu hình ---
INPUT_DIR = Path('data')
OUTPUT_DIR = Path('data_rembg_2')
SPLITS = ['Train_Alphabet', 'Test_Alphabet']
PADDING = 20

# Tạo folder đầu ra
for split in SPLITS:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

def process_file(in_path: Path, out_path: Path, padding: int):
    try:
        # 1. Load ảnh và remove background
        img = Image.open(in_path).convert("RGBA")
        img_no_bg = remove(img)  # PIL Image, mode RGBA

        # 2. Tính bounding box từ alpha channel
        arr = np.array(img_no_bg)
        alpha = arr[..., 3]
        ys, xs = np.where(alpha > 0)
        if len(xs) == 0 or len(ys) == 0:
            return f"Skipped (empty): {in_path}"

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        # padding
        x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
        x2 = min(x2 + padding, arr.shape[1])
        y2 = min(y2 + padding, arr.shape[0])

        # 3. Crop và chuyển nền trắng
        cropped = img_no_bg.crop((x1, y1, x2, y2))
        bg = Image.new("RGB", cropped.size, (255, 255, 255))
        bg.paste(cropped, mask=cropped.split()[3])

        # 4. Lưu ảnh
        bg.save(out_path, optimize=True)
        return f"Processed: {in_path}"
    except Exception as e:
        return f"Error {in_path}: {e}"

def main():
    # 1. Tạo danh sách tất cả các cặp (in_path, out_path)
    tasks = []
    for split in SPLITS:
        in_split = INPUT_DIR / split
        out_split = OUTPUT_DIR / split
        for cls in os.listdir(in_split):
            in_cls = in_split / cls
            out_cls = out_split / cls
            out_cls.mkdir(parents=True, exist_ok=True)
            for fname in os.listdir(in_cls):
                in_path = in_cls / fname
                out_path = out_cls / (in_path.stem + '.png')
                tasks.append((in_path, out_path))

    # 2. Chạy song song với ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, inp, outp, PADDING)
            for inp, outp in tasks
        ]
        for fut in as_completed(futures):
            print(fut.result())

if __name__ == '__main__':
    main()

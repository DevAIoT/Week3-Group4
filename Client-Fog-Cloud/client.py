import requests
import numpy as np
import time

FOG_URL = "http://localhost:5002/predict"
EDGE_THRESHOLD = 200  # ถ้า Variance ต่ำกว่านี้ จะทำเองที่เครื่อง Edge


def process_at_edge(image_data):
    """จำลองการทำ Inference ที่เครื่อง Edge"""
    # สำหรับ demo ใช้ค่าสุ่มแทน
    return {"top_confidence": 0.95, "inference_source": "Edge Locally"}


for i in range(50):
    # 1. จำลองหรือโหลดรูปภาพ (สลับ 3 ระดับความซับซ้อน)
    if i % 5 == 0:  # ภาพซับซ้อนมาก -> variance สูง (ส่ง Cloud)
        sample_img = (
            np.random.rand(1, 224, 224, 3).astype("float32") * 255
        )  # variance ~5400
    elif i % 3 == 0:  # ภาพซับซ้อนปานกลาง -> variance กลาง (ส่ง Fog)
        sample_img = (
            np.random.rand(1, 224, 224, 3).astype("float32") * 50
        )  # variance ~200-400
    else:  # ภาพธรรมดา -> variance ต่ำ (ทำที่ Edge)
        sample_img = np.random.rand(1, 224, 224, 3).astype("float32")  # variance ~0.08
    pixel_variance = np.var(sample_img)

    start_time = time.time()

    if pixel_variance < EDGE_THRESHOLD:
        # ทำงานที่เครื่องตัวเอง
        result = process_at_edge(sample_img)
    else:
        # ส่งไปให้ Fog
        response = requests.post(FOG_URL, json={"image": sample_img.tolist()})
        result = response.json()

    latency = time.time() - start_time
    print(f"Req {i+1}: Source={result['inference_source']}, Latency={latency:.4f}s")

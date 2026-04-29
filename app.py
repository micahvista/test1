import os
# 1. 环境变量配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import io
import tempfile
import torch
import librosa
import numpy as np
import base64
from PIL import Image
from scipy.spatial.distance import cosine
from transformers import Wav2Vec2Processor, Wav2Vec2Model, pipeline
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# 2. 初始化 FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 加载音频模型 (原有)
print("正在加载预训练模型 Wav2Vec 2.0...")
MODEL_NAME = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME, use_safetensors=True)
model.eval()
print("音频模型加载完成！")

# --- 新增：加载吸烟检测图像模型 ---
print("正在下载并加载吸烟检测图像模型 (首次运行可能需要几分钟)...")
# 这里我们使用 Hugging Face 社区开源的吸烟检测 ViT 模型
# pipeline 会自动下载并缓存模型权重
SMOKING_MODEL_NAME = "dima806/smoking_image_detection" 
SMOKING_MODEL_NAME = "dima806/smoker_image_classification"
smoking_pipeline = pipeline("image-classification", model=SMOKING_MODEL_NAME)
print("吸烟检测模型加载完成！")
# ---------------------------------

# 4. 音频特征提取函数 (原有)
def extract_deep_embedding(audio_bytes):
    try:
        current_device = next(model.parameters()).device
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        audio_input, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
        os.remove(temp_audio_path)

        inputs = processor(audio_input, sampling_rate=16000)
        input_tensor = torch.tensor(inputs['input_values'], dtype=torch.float32).to(current_device)

        with torch.no_grad():
            outputs = model(input_values=input_tensor)

        last_hidden_states = outputs.last_hidden_state
        embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"处理音频时出错: {e}")
        return None

# 5. 音频比对 API (原有)
@app.post("/api/compare_audio")
async def compare_audio(
    reference_audio: UploadFile = File(...),
    test_audio: UploadFile = File(...),
    threshold: float = Form(0.25)
):
    ref_bytes = await reference_audio.read()
    test_bytes = await test_audio.read()

    ref_embedding = extract_deep_embedding(ref_bytes)
    query_embedding = extract_deep_embedding(test_bytes)

    if ref_embedding is None or query_embedding is None:
        return {"status": "error", "message": "特征提取失败"}

    distance = cosine(ref_embedding, query_embedding)
    is_same = bool(distance < threshold)

    return {
        "status": "success",
        "distance": round(distance, 4),
        "threshold": threshold,
        "is_same": is_same,
        "message": "✅ 判定为同一类声音！" if is_same else "❌ 判定为不同类声音。"
    }

# --- 新增：实时视频流处理 WebSocket API ---
@app.websocket("/ws/detect_smoking")
async def detect_smoking_ws(websocket: WebSocket):
    await websocket.accept()
    print("前端已连接到吸烟检测 WebSocket")
    try:
        while True:
            # 接收前端发来的 base64 图像帧
            data = await websocket.receive_text()
            
            # 分离前缀并解码 base64
            header, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # 使用模型进行推理
            results = smoking_pipeline(image)
            
            # 解析推理结果
            # 模型通常返回包含 'smoking' 或 'not_smoking' 标签的字典列表
            is_smoking = False
            confidence = 0.0
            for res in results:
                label = res['label'].lower()
                # 兼容不同的标签命名，只要包含 smok 且不包含 not 且置信度大于 50% 就报警
                if 'smok' in label and 'not' not in label and res['score'] > 0.50:
                    is_smoking = True
                    confidence = res['score']
                    break
            
            # 将结果发回前端
            await websocket.send_json({
                "is_smoking": is_smoking,
                "confidence": float(confidence)
            })

    except WebSocketDisconnect:
        print("吸烟检测 WebSocket 客户端断开连接")
    except Exception as e:
        print(f"处理视频帧时出错: {e}")
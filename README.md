# Vision-Pro (Mark9)

**Edge-First AI Vision Assistant for the Visually Impaired**  
**Branch:** Mark9  
**Architecture:** Dual YOLO (COCO + Custom) + BLIP Captioning + LLaMA 3.2 (1B)  
**Platform:** Windows (tested), Linux (expected)

---

## 1️⃣ Project Overview

Vision-Pro is a real-time assistive vision system designed for visually impaired users.  
It prioritizes **low latency**, **stability**, and **calm, concise audio feedback**.

### Core Capabilities

- 🎥 Live webcam input  
- 🧠 **Dual YOLO inference**
  - General COCO objects
  - Custom-trained indoor objects
- 🖼️ Scene understanding via **BLIP image captioning**
- 🗣️ Natural-language audio output using a **local LLM**
- ⏱️ Temporal confidence & trigger control (no spam, no lag)
- 🔇 Anti-silence assurance speech

---

## 2️⃣ Architecture (Mark9 – Option C)

Webcam
↓
Dual YOLO Sentinel (runs every frame)
↓
Temporal Confidence Accumulator
↓
Semantic Trigger
↓
BLIP Captioning (on trigger only)
↓
LLaMA 3.2 (1B) via Ollama
↓
Windows Native TTS


### Why this architecture?

- **YOLO** → fast, reliable object detection  
- **BLIP** → lightweight scene grounding (avoids heavy DINO latency)  
- **LLM** → converts perception into human-friendly speech  
- Camera loop is **never blocked**

---

## 3️⃣ Hardware Requirements

### ✅ Minimum (CPU-only)

| Component | Requirement |
|---------|------------|
| CPU | Intel i5 / Ryzen 5 or better |
| RAM | 16 GB |
| GPU | ❌ Not required |
| Storage | 15 GB free |
| OS | Windows 10/11 (64-bit) |

⚠️ CPU-only mode works, but BLIP inference may take **2–4 seconds per trigger**.

---

### 🚀 Recommended (GPU-accelerated)

| Component | Requirement |
|---------|------------|
| GPU | NVIDIA RTX 3050 (6 GB) or better |
| CUDA | 12.1 |
| RAM | 16–32 GB |
| CPU | Intel i7 / Ryzen 7 |
| OS | Windows 10/11 |

ℹ️ GPU accelerates **YOLO only**.  
BLIP is intentionally forced to **CPU FP32** for stability.

---

## 4️⃣ Software Requirements

### Python
- **Python 3.12.x**

### CUDA (if GPU present)
- CUDA Toolkit **12.1**
- NVIDIA Driver **≥ 531.xx**

Verify GPU setup:
```bash
nvidia-smi

5️⃣ Python Dependencies (Exact Versions)

Create a virtual environment (recommended):
python -m venv venv
venv\Scripts\activate

INSTALL DEPENDENCIES
pip install \
ultralytics==8.3.251 \
torch==2.5.1+cu121 \
torchvision==0.20.1+cu121 \
torchaudio==2.5.1+cu121 \
opencv-python==4.12.0.88 \
numpy==2.2.0 \
pillow==10.4.0 \
transformers==4.48.2 \
sentencepiece==0.2.0 \
ollama==0.3.3

⚠️ Do NOT upgrade Torch beyond 2.5.1
BLIP currently requires non-safetensor fallback compatibility.

6️⃣ LLM Setup (Required)

Vision-Pro uses a local LLM via Ollama.

Install Ollama

👉 https://ollama.com/download

Pull the model
ollama pull llama3.2:1b

Verify
ollama list

7️⃣ YOLO Models
General YOLO (COCO)

Automatically downloaded:
yolov8n.pt

Custom YOLO (Indoor Objects)

You must provide:
weights/yolo_custom.pt

Example:
D:\Vision-Pro\weights\yolo_custom.pt

Custom classes include:
book, cell phone, remote, toothbrush, pen, table,
stairs, wallet, watch, ...

8️⃣ Running the Project

From the project root:
python main.py

Controls

📹 Webcam opens automatically

🗣️ Audio feedback plays through system speakers

❌ Press q to exit

9️⃣ Expected Runtime Behavior
Speech Characteristics (Mark9)

✅ One short sentence per event

✅ No introductions or meta explanations

✅ No hallucinated actions

✅ Calm, factual, demo-safe output

Example outputs:

“A person is holding a cell phone.”

“A book is in front of you.”

“One person is in front of you.”

🔟 Known Limitations (Intentional)

❌ No object counting yet (“two books”)

❌ No spatial relations (“to your left”)

❌ BLIP captions may lag ~2 seconds (by design)

❌ No cloud services (fully offline)

These are planned improvements, not bugs.

1️⃣1️⃣ Branch Information

this readme corresponds to Mark9

Previous Experimental Branches

Mark7 – Single YOLO + BLIP + DINO

Mark8 – Single YOLO + DINO + BLIP (dynamic switching)

Mark9 – Dual YOLO + BLIP (stable, demo-ready)



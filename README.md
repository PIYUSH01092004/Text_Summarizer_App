# 📝 Text Summarizer — AI-Powered Content Condensation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-F7931E?style=for-the-badge&logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](#)
[![API Docs](https://img.shields.io/badge/API_Docs-Swagger-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)](#)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

<p align="center">
  <b>A Full-Stack AI Web Application Powered by FastAPI & HuggingFace T5</b><br>
  Instantly condense long articles, dialogues, and paragraphs into concise, readable summaries.
</p>

</div>

---

## 📖 Executive Summary

**Text Summarizer** is an end-to-end AI web application designed to help users quickly digest large amounts of text. By utilizing a pre-trained **T5 (Text-to-Text Transfer Transformer)** model, this application intelligently analyzes dialogues and long-form content, extracting the most vital information and presenting it in a short, crisp summary.

The system is deployed as a production-ready **FastAPI** web server that handles tokenization, inference, and response formatting, accompanied by a stunning, modern glassmorphic web dashboard built with HTML, CSS, and JavaScript.

---

## ✨ Key Features

### 🤖 **1. Advanced Natural Language Processing**
- Powered by a fine-tuned **T5 Transformer model** from HuggingFace.
- Seamlessly handles dialogue-based text and long paragraphs (max 512 tokens).
- Fast and accurate text generation utilizing PyTorch tensor computations.

### ⚡ **2. High-Performance FastAPI Backend**
- FastAPI asynchronous endpoint (`POST /summarize/`) with automatic request validation using **Pydantic**.
- Integrated text-cleaning pipeline to strip HTML tags, extra spaces, and newlines before processing.
- Model device auto-selection (CUDA, MPS, or CPU) for optimal performance.

### 🎨 **3. Glassmorphic Interactive Dashboard**
- Modern dark-mode aesthetic built with pure CSS variables, backdrop blurs, dynamic glow accents, and floating orb background animations.
- Real-time loading states and smooth CSS transitions.
- One-click "Copy to Clipboard" functionality for the generated summary.

### 🚀 **4. Production Ready**
- Built-in static file rendering using `Jinja2Templates`.
- Zero frontend build-step required; everything is optimized in a single responsive HTML view.

---

## ⚙️ Architecture

### **Backend Pipeline**
1. **Input Validation**: FastAPI + Pydantic schema ensures valid dialogue strings.
2. **Text Preprocessing**: Regex-based cleaning (removes HTML, normalizes spaces).
3. **Tokenization**: `T5Tokenizer` converts raw text into padded, truncated tensor IDs.
4. **Inference**: `T5ForConditionalGeneration` model generates summary tokens using beam search (`num_beams=4`).
5. **Decoding**: Tokens are converted back to human-readable text and returned as a JSON response.

### **Frontend Pipeline**
- **UI Render**: `Jinja2Templates` serves `index.html` on the root (`/`) route.
- **Client Action**: JavaScript intercepts form submission and initiates a `fetch` POST request.
- **State Management**: Dynamic UI updates (spinners, disabling buttons, error catching, and success animations).

---

## 🚀 Deployment (Render.com)

This project is fully configured to be deployed on **Render.com** as a Web Service.

1. **Root Directory**: `save_summary_model`
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

## 💻 Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/PIYUSH01092004/Text_Summarizer_App.git
cd Text_Summarizer_App

# 2. Install dependencies
pip install -r save_summary_model/requirements.txt

# 3. Navigate to the app directory
cd save_summary_model

# 4. Start the server
uvicorn app:app --reload --port 8000
```
Visit `http://localhost:8000` in your browser.

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="2.0",
)

# ---------------------------------------------------------------------------
# Model loading  — done ONCE at startup, synchronously so the server is
# only marked "ready" after the model is in memory.
# ---------------------------------------------------------------------------
MODEL_ID = "piyush01092004/text-summarizer-t5"

print(f"[startup] Loading model '{MODEL_ID}' ...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()  # inference-only mode — disables dropout, saves memory
print(f"[startup] Model ready on {device}.")

# Thread pool so CPU-bound inference doesn't block the async event loop
_executor = ThreadPoolExecutor(max_workers=2)

# Templating
templates = Jinja2Templates(directory=".")


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class DialogueInput(BaseModel):
    dialogue: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)   # Windows line-endings
    text = re.sub(r"\s+", " ", text)    # collapse whitespace
    text = re.sub(r"<.*?>", " ", text)  # strip HTML tags
    return text.strip().lower()


def _run_inference(dialogue: str) -> str:
    """Runs T5 inference synchronously — called inside a thread pool."""
    dialogue = clean_data(dialogue)

    # Tokenize — do NOT pad to max_length; let the model process only the
    # real tokens so inference is proportional to actual text length.
    inputs = tokenizer(
        dialogue,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # num_beams=2 is a good balance between quality and speed on CPU.
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    if not dialogue_input.dialogue.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Offload blocking inference to the thread pool so the event loop stays free
    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(
            _executor, _run_inference, dialogue_input.dialogue
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")

    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    """Health-check endpoint for Railway / uptime monitors."""
    return {"status": "ok", "device": str(device)}
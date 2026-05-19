from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Initialize our fastapi app
app=FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# model and tokenizer
model=T5ForConditionalGeneration.from_pretrained(".")
tokenizer=T5Tokenizer.from_pretrained(".")

# Device 
if torch.backends.mps.is_available():
    device=torch.device("mps")
elif torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

model.to(device)

# Templating 
templates=Jinja2Templates(directory=".")

# Input Schema for Dialogue => String
class DialogueInput(BaseModel):
    dialogue:str


def clean_data(text):
    text=re.sub(r"\r\n", " ", text) #Lines
    text=re.sub(r"\s+", " ", text) #space
    text=re.sub(r"<.*?>", " ", text) #HTML Tags
    text=text.strip().lower()
    return text

def summarize_dialogue(dialogue:str) -> str:
    dialogue = clean_data(dialogue) #Clean
    #Tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    #Generate the Summary => token_ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # Token_id convert to summary => Decode
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary


#API Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input:DialogueInput):
    summary= summarize_dialogue(dialogue_input.dialogue)
    return {"summary":summary}

@app.get("/",response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse(request=request, name="index.html")
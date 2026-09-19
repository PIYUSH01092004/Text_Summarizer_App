FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model at build time so cold starts are fast
RUN python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; \
    T5ForConditionalGeneration.from_pretrained('piyush01092004/text-summarizer-t5'); \
    T5Tokenizer.from_pretrained('piyush01092004/text-summarizer-t5'); \
    print('Model cached successfully!')"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

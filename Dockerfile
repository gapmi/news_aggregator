FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=120 \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1+cpu \
 && pip install --no-cache-dir --default-timeout=120 \
      -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
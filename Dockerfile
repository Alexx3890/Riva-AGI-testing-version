# Production image for the Riva real-time voice gateway.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN addgroup --system riva && adduser --system --ingroup riva riva

COPY requirements.txt ./requirements.txt
COPY voice_speech/requirements.txt ./voice_speech/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r voice_speech/requirements.txt

COPY . .
RUN chown -R riva:riva /app

USER riva
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "voice_speech.web_server:app", "--host", "0.0.0.0", "--port", "8000"]

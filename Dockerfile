FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir "accelerate>=1.2.0" "trl>=0.15.0" unsloth

ENV PYTHONPATH="/app/env:${PYTHONPATH}"
ENV SPACE_OUTPUT_ROOT="/tmp/adapt-space"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.11-slim

WORKDIR /app

# Cài torch CPU trước (tránh pip kéo bản CUDA nặng vài GB không cần thiết,
# vì service này chạy embedding + LLM client thuần CPU trong Docker Desktop).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]

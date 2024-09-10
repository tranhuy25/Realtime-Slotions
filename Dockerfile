# Dockerfile

# Chọn base image
FROM python:3.9-slim

# Cài đặt các thư viện cần thiết
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY src/ /app/src
COPY data/ /app/data

# Đặt entrypoint cho container
ENTRYPOINT ["python", "/app/src/main.py"]

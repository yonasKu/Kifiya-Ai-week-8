# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "src/serve_model.py"]

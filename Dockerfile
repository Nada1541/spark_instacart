FROM python:3.10-slim

# Java is required for PySpark
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre procps unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 1000 --retries 5 -r requirements.txt

# Keep the container alive for `docker compose exec` workflows
CMD ["tail", "-f", "/dev/null"]

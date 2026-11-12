FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-nodes.txt .
RUN pip install --no-cache-dir -r requirements-nodes.txt
COPY worker.py entrypoint.sh ./
# Corriger les fins de ligne et rendre le script exécutable
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh 
EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
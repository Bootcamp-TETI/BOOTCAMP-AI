FROM python:3.11-slim

WORKDIR /app

# Torch CPU dulu, terpisah — wheel default (CUDA) ~2.5GB dan tidak berguna di server tanpa GPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces mengekspos satu port (7860) -> Streamlit; FastAPI jalan internal di 8000
ENV API_URL=http://localhost:8000
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true"]

FROM python:3.12-slim
WORKDIR /app
ENV MODEL_PATH=/app/ids_best_model.pkl
ENV THRESHOLD_PATH=/app/threshold.json

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ids_api.py /app/ids_api.py
COPY ids_best_model.pkl /app/ids_best_model.pkl
COPY ids_xgboost_model.pkl /app/ids_xgboost_model.pkl
COPY threshold.json /app/threshold.json

EXPOSE 8000
CMD ["uvicorn", "ids_api:app", "--host", "0.0.0.0", "--port", "8000"]

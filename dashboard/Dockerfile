FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY ids_best_model.pkl /app/ids_best_model.pkl
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir reportlab==4.2.5

COPY ids_dashboard.py /app/ids_dashboard.py
COPY ids_xgboost_model.pkl /app/ids_xgboost_model.pkl
COPY ids_best_model.pkl /app/ids_best_model.pkl
EXPOSE 8501
CMD ["streamlit", "run", "ids_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
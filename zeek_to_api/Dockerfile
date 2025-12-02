FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY zeek_to_api.py /app/zeek_to_api.py

CMD ["python", "zeek_to_api.py"]
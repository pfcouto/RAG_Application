FROM python:3.10-slim

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY data/dataset.txt /app/data/

COPY .env.docker /app/.env

EXPOSE 8000

CMD ["python", "app/main.py"]
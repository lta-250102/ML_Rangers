FROM python:latest

WORKDIR /app

COPY requirements.txt .

RUN apt-get update

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]

EXPOSE 5040

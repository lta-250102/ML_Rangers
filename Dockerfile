FROM python:latest

COPY . /app

WORKDIR /app

RUN apt-get update

RUN pip install -r requirements.txt

CMD ["python", "src/main.py"]

EXPOSE 5040
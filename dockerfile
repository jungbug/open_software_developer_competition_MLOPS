FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

COPY . /app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

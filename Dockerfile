FROM python:3.11.1
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "requirements.txt", "./"]
RUN pip install -r requirements.txt
COPY . . 
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
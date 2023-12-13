FROM python:3.9

RUN  pip3 install --upgrade flask flask-wtf redis Flask-Debug requests minio psycopg2-binary

RUN  mkdir /app
COPY app /app
COPY ./init_tables.py /init_tables.py
EXPOSE	5000
WORKDIR /
CMD ["python3", "init_tables.py"]
CMD	FLASK_APP=app flask run -h 0.0.0.0

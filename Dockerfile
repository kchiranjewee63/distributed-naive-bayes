FROM openjdk:8-jre-alpine

RUN apk add --no-cache python3-dev && \
    python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    ln -s /usr/bin/python3 /usr/bin/python
RUN apk add --no-cache bash


RUN pip3 install pyspark


RUN wget -q https://dlcdn.apache.org/spark/spark-3.2.4/spark-3.2.4-bin-hadoop3.2.tgz && \
    tar -xzf spark-3.2.4-bin-hadoop3.2.tgz && \
    mv spark-3.2.4-bin-hadoop3.2 spark && \
    rm spark-3.2.4-bin-hadoop3.2.tgz


ENV SPARK_HOME /spark
ENV PATH $PATH:${SPARK_HOME}/bin

WORKDIR /src
COPY . .

# ENTRYPOINT ["python3", "main.py"]

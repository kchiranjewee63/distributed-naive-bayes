FROM openjdk:8-jre-alpine

RUN apk add --no-cache python3-dev && \
    python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    ln -s /usr/bin/python3 /usr/bin/python
RUN apk add --no-cache bash
RUN apk add --no-cache enchant-dev
RUN apk add --no-cache aspell-en

ENV SPARK_VERSION 3.2.4
ENV HADOOP_VERSION 3.2

RUN pip3 install pyspark==${SPARK_VERSION} nltk pyenchant


RUN wget -q https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz


ENV SPARK_HOME /spark
ENV PATH $PATH:${SPARK_HOME}/bin

WORKDIR /src
COPY . .

ENTRYPOINT ["python3", "main.py"]
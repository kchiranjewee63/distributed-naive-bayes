from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from multiNomialNaiveBayse import predict

def calAccuracy(data, parameters):
    predictor_udf = udf(lambda review: predict(review, parameters), StringType())
    predictions = data.withColumn('prediction', predictor_udf(data.review))
    accuracy = predictions.filter(predictions.sentiment == predictions.prediction).count()/data.count()
    return accuracy
    
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from multiNomialNaiveBayse import predict

def calAccuracy(data, parameters):
    predictor_udf = udf(lambda review: predict(review, parameters), StringType())
    predictions = data.select('sentiment', predictor_udf(data.review).alias('prediction'))
    accuracy = predictions.filter(predictions.sentiment == predictions.prediction).count()/predictions.count()
    return accuracy
    
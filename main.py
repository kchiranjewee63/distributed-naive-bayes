from pyspark.sql import SparkSession
from IMDBMovieReviewData import IMDBMovieReviewData
from multiNomialNaiveBayse import train
from modelEvaulation import calAccuracy


spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

data = IMDBMovieReviewData(spark, "IMDB Dataset.csv")
data.loadData()
data.preprocessData()
data.splitData()


parameters = train(data.training_data)




calAccuracy(data.training_data, parameters)
calAccuracy(data.testing_data, parameters)





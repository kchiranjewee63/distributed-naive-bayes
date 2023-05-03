from pyspark.sql import SparkSession
from IMDBMovieReviewData import IMDBMovieReviewData
from multiNomialNaiveBayse import train, calAccuracy
import json


spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()


print("Loading data...")
data = IMDBMovieReviewData(spark, "IMDB Dataset.csv")
data.loadData()

print("Preprocessing data...")
data.preprocessData()

print("Splitting data into training and testing set...")
data.splitData()


print("Training multinomial naive bayse on IMBD data...")
parameters = train(data.training_data)

print("Writing learned parameters to a json file...")
json_parameters = json.dumps(parameters)
with open("parameters.json", "w") as outfile:
    outfile.write(json_parameters)


# print("Calculating train accuracy...")
# print(f"Train Accuracy: {calAccuracy(data.training_data, parameters)}")

print("Calculating test accuracy...")
print(f"Test Accuracy: {calAccuracy(data.testing_data, parameters)}")

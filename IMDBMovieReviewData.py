from pyspark.sql.functions import lower, regexp_replace, split, concat_ws

class IMDBMovieReviewData:
    def __init__(self, spark_session, data_path):
        self.spark_session = spark_session
        self.data_path = data_path
    
    def loadData(self):
        self.movie_reviews_df = self.spark_session.read.csv(self.data_path, header=True, inferSchema=True, quote='"', escape='"')

    def preprocessData(self):
        self.movie_reviews_df = self.movie_reviews_df \
            .withColumn('review', lower("review")) \
            .withColumn('review', regexp_replace("review", "[^a-zA-Z0-9\\s]", ""))

    def splitData(self, training_fraction = 0.8, testing_fraction = 0.2):
        self.training_data, self.testing_data = self.movie_reviews_df.randomSplit([training_fraction, testing_fraction])
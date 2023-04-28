class IMDBMovieReviewData:
    def __init__(self, spark_session, data_path):
        self.spark_session = spark_session
        self.data_path = data_path
    
    def loadData(self):
        self.movie_reviews_df = self.spark_session.read.csv(self.data_path, header=True, inferSchema=True, quote='"', escape='"')
        
    def preprocessData(self):
#         ToDO
#         1. Convert words to lowercase 
#         2. Remove special characters 
#         3. Remove stop words 
#         3. Stemming
        pass
        
    def splitData(self, training_fraction = 0.7, testing_fraction = 0.3):
        self.training_data, self.testing_data = self.movie_reviews_df.randomSplit([training_fraction, testing_fraction])
        self.movie_reviews_df.unpersist()
        del self.movie_reviews_df
from pyspark.sql.functions import lower, regexp_replace, split, concat_ws
from pyspark.sql.types import StringType,ArrayType
from pyspark.sql.functions import udf
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

class IMDBMovieReviewData:
    def __init__(self, spark_session, data_path):
        self.spark_session = spark_session
        self.data_path = data_path
    
    def loadData(self):
        self.movie_reviews_df = self.spark_session.read.csv(self.data_path, header=True, inferSchema=True, quote='"', escape='"')

    def preprocessData(self):
        # Convert words to lowercase
        self.movie_reviews_df = self.movie_reviews_df.withColumn("review", lower(self.movie_reviews_df.review))

        # Remove special characters
        self.movie_reviews_df = self.movie_reviews_df.withColumn("review", regexp_replace(self.movie_reviews_df.review, "[^a-zA-Z0-9\\s]", ""))

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        self.movie_reviews_df = self.movie_reviews_df.withColumn("words", split(self.movie_reviews_df.review, "\\s+")).drop("review")
        remove_stop_words_udf = udf(lambda review: [word for word in review if word not in stop_words], ArrayType(StringType()))
        self.movie_reviews_df = self.movie_reviews_df.withColumn("nonStopWords", remove_stop_words_udf(self.movie_reviews_df.words)).drop("words")

        # Stemming
        stemmer = SnowballStemmer(language='english')
        stem_words_udf = udf(lambda review: " ".join([stemmer.stem(word) for word in review]))
        self.movie_reviews_df = self.movie_reviews_df.withColumn("review", stem_words_udf(self.movie_reviews_df.nonStopWords)).drop("nonStopWords")

    def splitData(self, training_fraction = 0.7, testing_fraction = 0.3):
        self.training_data, self.testing_data = self.movie_reviews_df.randomSplit([training_fraction, testing_fraction])
        self.movie_reviews_df.unpersist()
        del self.movie_reviews_df
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
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer(language='english')

        remove_stop_words_udf = udf(lambda x: [word for word in x if word not in stop_words], ArrayType(StringType()))
        stem_udf = udf(lambda x: ' '.join([stemmer.stem(word) for word in x]), StringType())

        self.movie_reviews_df = self.movie_reviews_df \
            .withColumn('review', lower("review")) \
            .withColumn('review', regexp_replace("review", "[^a-zA-Z0-9\\s]", "")) \
            .withColumn('review', split("review", "\\s+")) \
            .withColumn('review', remove_stop_words_udf('review')) \
            .withColumn('review', stem_udf('review'))

    def splitData(self, training_fraction = 0.7, testing_fraction = 0.3):
        self.training_data, self.testing_data = self.movie_reviews_df.randomSplit([training_fraction, testing_fraction])
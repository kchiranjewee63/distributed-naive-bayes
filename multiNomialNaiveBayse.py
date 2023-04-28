from pyspark.sql.functions import split, explode
from functools import reduce
from pyspark.sql.functions import sum as _sum
from math import log

LAPLACE_SMOOTHING_PARAMETER = 1

def calPriorProbs(data):
    counts = data.groupBy("sentiment").count().rdd.collectAsMap()
    total = counts["positive"] + counts["negative"]
    return counts["positive"]/total, counts["negative"]/total
    
def countWordsInAClass(reviews_data_frame, class_label):
    class_reviews = reviews_data_frame.filter(reviews_data_frame.sentiment == class_label)
    words_column = explode(split(class_reviews.review, "\s+")).alias("word")
    words_counts = class_reviews.select(words_column).groupBy("word").count()
    total_count = words_counts.agg(_sum("count")).collect()[0][0]
    words_counts = words_counts.rdd.collectAsMap()
    return {"total-count":total_count, "words-counts":words_counts}
    
def train(data):
    pos_prior_prob, neg_prior_prob = calPriorProbs(data)
    pos_counts = countWordsInAClass(data, "positive")
    neg_counts = countWordsInAClass(data, "negative")
    parameters = {
        "pos-prior-prob":pos_prior_prob,
        "neg-prior-prob":neg_prior_prob,
        "pos-counts":pos_counts,
        "neg-counts":neg_counts}
    return parameters

def calLogProb(review, class_counts, class_prior_prob):
    words = review.split()
    probs_list = [class_counts["words-counts"].get(word, LAPLACE_SMOOTHING_PARAMETER)/(class_counts["total-count"] + LAPLACE_SMOOTHING_PARAMETER*len(class_counts["words-counts"])) for word in words]
    return log(class_prior_prob) + reduce(lambda a, b: a + log(b), probs_list, 0)

def predict(review, parameters):
    pos_log_prob = calLogProb(review, parameters["pos-counts"], parameters["pos-prior-prob"])
    neg_log_prob = calLogProb(review, parameters["neg-counts"], parameters["neg-prior-prob"])
    return "positive" if pos_log_prob > neg_log_prob else "negative"
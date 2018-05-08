import pyspark as ps
import numpy as np

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

import utils
import pipeline
import model

if __name__ == '__main__':

    spark = (
            ps.sql.SparkSession.builder
            .master("local[4]")
            .appName("jampoq")
            .getOrCreate()
            )

    sc = spark.sparkContext

    # Read in data
    cell_accessories = 's3a://capstone-g65ds/data/reviews_Cell_Phones_and_Accessories_5.json'
    toys_games = 's3a://capstone-g65ds/data/reviews_Toys_and_Games_5.json'
    kindle = 's3a://capstone-g65ds/data/reviews_Kindle_Store_5.json'
    cds_vinyl = 's3a://capstone-g65ds/data/reviews_CDs_and_Vinyl_5.json'
    books = 's3a://capstone-g65ds/data/reviews_Books_5.json'

    kindle_local = """file:///home/hadoop/reviews_Kindle_Store_5.json"""

    df = spark.read.json(cds_vinyl)

    ###############################
    # Model scores for TFIDF only.
    ###############################

    p = pipeline.DataPipeline(df,spark,sc)
    p.get_data(10,'.85')
    p.add_first_layer_features()
    p.add_sec_layer_features()
    #p.add_sentiment()
    p.add_pyspark_features(transform_type='tfidf',
                           pca=False,
                           pca_k = 500,
                           chi_sqr = False,
                           chi_feature_num = 500)

    m = model.Model(p.df,.7,'results_tfidf_only.txt')
    m.run_logistic_cv()
    m.run_linear_svc()
    m.run_gradient_boost()

    ###############################
    # Model scores for TFIDF + PCA
    ###############################

    p2 = pipeline.DataPipeline(df,spark,sc)
    p2.get_data(10,'.85')
    p2.add_first_layer_features()
    p2.add_sec_layer_features()
    #p2.add_sentiment()
    p2.add_pyspark_features(transform_type='tfidf',
                            pca=True,
                            pca_k = 500,
                            chi_sqr = False,
                            chi_feature_num = 500)

    m2 = model.Model(p2.df,.7,'results_tfidf_pca.txt')
    m2.run_logistic_cv()
    #m2.run_linear_svc()
    m2.run_gradient_boost()

    ###############################
    # Model scores for TFIDF + Chi
    ###############################

    p3 = pipeline.DataPipeline(df,spark,sc)
    p3.get_data(10,'.85')
    p3.add_first_layer_features()
    p3.add_sec_layer_features()
    #p3.add_sentiment()
    p3.add_pyspark_features(transform_type='tfidf',
                            pca=False,
                            pca_k = 500,
                            chi_sqr = True,
                            chi_feature_num = 500)

    m3 = model.Model(p3.df,.7,'results_tfidf_chi.txt')
    m3.run_logistic_cv()
    #m3.run_linear_svc()
    m3.run_gradient_boost()

    ###################################
    # Model scores for CountVect only
    ###################################

    p4 = pipeline.DataPipeline(df,spark,sc)
    p4.get_data(10,'.85')
    p4.add_first_layer_features()
    p4.add_sec_layer_features()
    #p4.add_sentiment()
    p4.add_pyspark_features(transform_type='countvectorizer',
                            pca=False,
                            pca_k = 500,
                            chi_sqr = False,
                            chi_feature_num = 500)

    m4 = model.Model(p4.df,.7,'results_countvect_only.txt')
    m4.run_logistic_cv()
    #m4.run_linear_svc()
    m4.run_gradient_boost()

    ###################################
    # Model scores for CountVect + PCA
    ###################################

    p5 = pipeline.DataPipeline(df,spark,sc)
    p5.get_data(10,'.85')
    p5.add_first_layer_features()
    p5.add_sec_layer_features()
    #p5.add_sentiment()
    p5.add_pyspark_features(transform_type='countvectorizer',
                            pca=True,
                            pca_k = 500,
                            chi_sqr = False,
                            chi_feature_num = 500)

    m5 = model.Model(p5.df,.7,'results_countvect_pca.txt')
    m5.run_logistic_cv()
    #m5.run_linear_svc()
    m5.run_gradient_boost()

    ###################################
    # Model scores for CountVect + Chi
    ###################################

    p6 = pipeline.DataPipeline(df,spark,sc)
    p6.get_data(10,'.85')
    p6.add_first_layer_features()
    p6.add_sec_layer_features()
    #p6.add_sentiment()
    p6.add_pyspark_features(transform_type='countvectorizer',
                            pca=False,
                            pca_k = 500,
                            chi_sqr = True,
                            chi_feature_num = 500)

    m6 = model.Model(p6.df,.7,'results_countvect_chi.txt')
    m6.run_logistic_cv()
    #m6.run_linear_svc()
    m6.run_gradient_boost()

    ################################
    # Model scores for Ngrams only
    ################################

    p7 = pipeline.DataPipeline(df,spark,sc)
    p7.get_data(10,'.85')
    p7.add_first_layer_features()
    p7.add_sec_layer_features()
    #p7.add_sentiment()
    p7.add_pyspark_features(transform_type='bigram',
                            pca=False,
                            pca_k = 500,
                            chi_sqr = False,
                            chi_feature_num = 500)

    m7 = model.Model(p7.df,.7,'results_ngrams_only.txt')
    m7.run_logistic_cv()
    #m7.run_linear_svc()
    m7.run_gradient_boost()

    ################################
    # Model scores for Ngrams + PCA
    ################################

    p8 = pipeline.DataPipeline(df,spark,sc)
    p8.get_data(10,'.85')
    p8.add_first_layer_features()
    p8.add_sec_layer_features()
    #p8.add_sentiment()
    p8.add_pyspark_features(transform_type='bigram',
                            pca=True,
                            pca_k = 500,
                            chi_sqr = False,
                            chi_feature_num = 500)

    m8 = model.Model(p8.df,.7,'results_ngrams_pca.txt')
    m8.run_logistic_cv()
    #m8.run_linear_svc()
    m8.run_gradient_boost()

    ################################
    # Model scores for Ngrams + Chi
    ################################

    p9 = pipeline.DataPipeline(df,spark,sc)
    p9.get_data(10,'.85')
    p9.add_first_layer_features()
    p9.add_sec_layer_features()
    #p9.add_sentiment()
    p9.add_pyspark_features(transform_type='bigram',
                           pca=False,
                           pca_k = 500,
                           chi_sqr = True,
                           chi_feature_num = 500)

    m9 = model.Model(p9.df,.7,'results_ngrams_chi.txt')
    m9.run_logistic_cv()
    #m9.run_linear_svc()
    m9.run_gradient_boost()

    ##################################
    # Model scores for word2vec only
    ##################################

    p10 = pipeline.DataPipeline(df,spark,sc)
    p10.get_data(10,'.85')
    p10.add_first_layer_features()
    p10.add_sec_layer_features()
    #p10.add_sentiment()
    p10.add_pyspark_features(transform_type='word2vec',
                           pca=False,
                           pca_k = 500,
                           chi_sqr = False,
                           chi_feature_num = 500)

    m10 = model.Model(p10.df,.7,'results_word2vec_only.txt')
    m10.run_logistic_cv()
    #m10.run_linear_svc()
    m10.run_gradient_boost()

    ##################################
    # Model scores for word2vec + PCA
    ##################################

    p11 = pipeline.DataPipeline(df,spark,sc)
    p11.get_data(10,'.85')
    p11.add_first_layer_features()
    p11.add_sec_layer_features()
    #p11.add_sentiment()
    p11.add_pyspark_features(transform_type='word2vec',
                           pca=True,
                           pca_k = 500,
                           chi_sqr = False,
                           chi_feature_num = 500)

    m11 = model.Model(p11.df,.7,'results_word2vec_pca.txt')
    m11.run_logistic_cv()
    #m11.run_linear_svc()
    m11.run_gradient_boost()

    ##################################
    # Model scores for word2vec + Chi
    ##################################

    p12 = pipeline.DataPipeline(df,spark,sc)
    p12.get_data(10,'.85')
    p12.add_first_layer_features()
    p12.add_sec_layer_features()
    #p12.add_sentiment()
    p12.add_pyspark_features(transform_type='word2vec',
                           pca=False,
                           pca_k = 500,
                           chi_sqr = True,
                           chi_feature_num = 500)

    m12 = model.Model(p12.df,.7,'results_word2vec_chi.txt')
    m12.run_logistic_cv()
    #m12.run_linear_svc()
    m12.run_gradient_boost()

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, udf, coalesce
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import (RegexTokenizer,
                                StopWordsRemover,
                                CountVectorizer,
                                OneHotEncoder,
                                StringIndexer,
                                VectorAssembler,
                                NGram,
                                Word2Vec,
                                StandardScaler,
                                HashingTF,
                                IDF,
                                PCA)

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import utils

##############
# Pipeline
##############

class DataPipeline():

    def __init__(self, pyspark_df, spark_session, sc):
        self.df = pyspark_df
        self.session = spark_session
        self.sc = sc

    def add_helpful_col(self):
        '''
        Takes in a pyspark dataframe and returns a dataframe with
        the helpfulness columns added.

        Input:
        --------
        None

        Output:
        --------
        None
        '''

        # Creates two new columns from the helpful column.
        self.df = self.df.withColumn('review_count',utils.ith('helpful',lit(1))) \
                         .withColumn('helpful_count',utils.ith('helpful',lit(0)))

    def filter_helpful(self,n):
        '''
        Filter for all reviews where it's been tagged at least
        n times.

        Input:
        --------
        n : How many minimum tags the review has
        Output:
        --------
        None
        '''
        self.add_helpful_col()
        self.df = self.df.filter(self.df['review_count'] >= n) \
                         .withColumn('helpfulness',coalesce(self.df['helpful_count']/self.df['review_count'],lit(0)))

    def get_data(self,n,threshold):
        '''
        Get data to add features to and put into models.

        Input:
        --------
        n : How many minum tages the review has
        threshold : ratio of helpful not helpful you want

        Output:
        --------
        None
        '''
        self.filter_helpful(n)

        # Store dataframe into temp table
        self.df.registerTempTable('reviews')

        # Spark dataframe that will get vectorized and put in a model.
        self.df = self.session.sql("""select
                                      reviewerID,
                                      overall,
                                      reviewText,
                                      unixReviewTime,
                                      case when helpfulness >= {thresh} then 1
                                           else 0
                                      end as label
                                      from reviews
                               """.format(thresh = str(threshold)))

        return self.df

    def add_first_layer_features(self):
        '''
        Add first layer of features using the udf functions from util.py.

        Input:
        -------
        None

        Output:
        -------
        None
        '''
        self.df = self.df.withColumn('sentence_cnt',utils.sentence_count(self.df.reviewText)) \
                         .withColumn('word_cnt',utils.word_count(self.df.reviewText)) \
                         .withColumn('capital_cnt',utils.count_capital(self.df.reviewText)) \
                         .withColumn('upper_word_cnt',utils.all_caps(self.df.reviewText)) \
                         .withColumn('punctuation_cnt',utils.count_punctuation(self.df.reviewText)) \
                         .withColumn('overall_transform',utils.overall_transform(self.df.overall))

    def add_sec_layer_features(self):
        '''
        Add second layer of features using features from the first layer.

        Input:
        -------
        None

        Output:
        -------
        None
        '''
        self.df = self.df.withColumn('avg_word_cnt',self.df.word_cnt/self.df.sentence_cnt) \
                         .withColumn('avg_punc_cnt',self.df.punctuation_cnt/self.df.sentence_cnt) \
                         .withColumn('avg_capital_cnt',self.df.capital_cnt/self.df.sentence_cnt) \
                         .withColumn('avg_upper_cnt',self.df.upper_word_cnt/self.df.sentence_cnt)

    def add_sentiment(self,review_col='reviewText',label_col='label'):
        '''
        Add sentiment columns for each each review in pyspark dataframe.

        Sentiments:
        ------------
        neg = negative
        neu = neutral
        pos = positive
        compound = compound score of neg, neu and pos

        Input:
        -------
        df_pyspark : pyspark dataframe
        reviewText : str
        label_col : int

        Output:
        -------
        None
        '''
        df = self.df.toPandas()

        vader_matrix = np.empty((df.shape[0],4))

        analyser = SentimentIntensityAnalyzer()

        for i, review in enumerate(df[review_col]):
            sentiments = analyser.polarity_scores(review)
            vader_matrix[[i],[0]] = sentiments['neg']
            vader_matrix[[i],[1]] = sentiments['neu']
            vader_matrix[[i],[2]] = sentiments['pos']
            vader_matrix[[i],[3]] = sentiments['compound']

        df_vader = pd.DataFrame(vader_matrix)
        df_vader.columns = ['neg','neu','pos','compound']

        df_final = pd.concat([df,df_vader],axis=1)

        # Convert pandas dataframe back into pyspark dataframe.
        sqlCtx = SQLContext(self.sc)
        self.df = sqlCtx.createDataFrame(data=df_final)

    def add_pyspark_features(self,
                             transform_type='countvectorizer',
                             pca=False,
                             pca_k = 3000):
        '''
        Add built in pyspark feature transformations using pyspark's
        Pipeline.

        Input:
        -------
        transform_type : str (Determines how to transform the reviews
                               - 'countvectorizer'
                               - 'bigram'
                               - 'tfidf'
                               - 'word2vec')
        pca : boolean (Determines whether to run PCA on the tranformed review.)
        pca_number : int (Number of features you want to reduce to.)

        Output:
        -------
        None
        '''

        # Set up stages.
        stages = []

        # Tokenize reviews into vectors of words.
        regexTokenizer = RegexTokenizer(inputCol="reviewText",
                                        outputCol="words",
                                        pattern="\\W")
        # Add to stages.
        stages += [regexTokenizer]

        # Remove stopwords from the word vectors.
        add_stopwords = ['the','a','to']
        stopwordsRemover = StopWordsRemover(inputCol="words",
                                            outputCol="filtered").setStopWords(add_stopwords)
        # Add to stages.
        stages += [stopwordsRemover]


        # Using CountVectorizer as our review transformation.
        if transform_type == 'countvectorizer':
            # Create count vectors from the filtered bag of words.
            countVectors = CountVectorizer(inputCol="filtered",
                                           outputCol="review_vector",
                                           vocabSize=5000,
                                           minDF=5)
            # Add to stages.
            stages += [countVectors]

        # Using TFIDF as our review transformation.
        if transform_type == 'tfidf':
            # Creating IDF from the words the filtered words
            hashingTF = HashingTF(inputCol="filtered",
                                  outputCol="rawFeatures",
                                  numFeatures=5000)
            idf = IDF(inputCol="rawFeatures",
                      outputCol="review_vector",
                      minDocFreq=5)
            # Add to stages
            stages += [hashingTF,idf]

        # Using bigrams as our review transformation.
        if transform_type == 'bigram':

            # Single grams.
            unigram = NGram(n=1, inputCol='words', outputCol='unigrams')
            stages +=[unigram]

            # Add n-grams to feature set.
            bigrams = NGram(n=2, inputCol="words", outputCol="bigrams")
            stages += [bigrams]

            # Vectorize unigrams
            unigrams_vector = CountVectorizer(inputCol="unigrams",
                                              outputCol="unigrams_vector",
                                              vocabSize=2500)
            stages += [unigrams_vector]

            bigrams_vector = CountVectorizer(inputCol="bigrams",
                                             outputCol="bigrams_vector",
                                             vocabSize=2500)
            stages += [bigrams_vector]

            # Vector assemble the unigrams and the bigrams
            ngrams = VectorAssembler(inputCols=['unigrams_vector','bigrams_vector'],
                                     outputCol='review_vector')
            stages += [ngrams]

        # Using word2vec as our review transformation.
        if transform_type == 'word2vec':
            word2vec = Word2Vec(vectorSize=5000,
                                minCount=0,
                                inputCol="words",
                                outputCol="review_vector")
            stages += [word2vec]

        # Use PCA if user wants to use it.
        if pca:
            pca = PCA(k=pca_k, inputCol="review_vector", outputCol="pcaFeatures")
            stages += [pca]

        # Perform one hot encoding on all categorical variables.
        categorical_cols = ['reviewerID']

        for col in categorical_cols:
            # Map each categorical value to an index (number).
            stringIndexer = StringIndexer(inputCol=col,
                                          outputCol=col + "_Index")
            # Use OneHotEncoder to convert categorical variables
            # into binary SparseVectors. Similar to pd.get_dummies()
            encoder = OneHotEncoder(inputCol=stringIndexer.getOutputCol(),
                                    outputCol=col + "_classVec")
            # Add to stages
            stages += [stringIndexer, encoder]

        # Numeric columns
        numericCols = ['overall_transform']

        # Get columns that we want from before spark pipleline.
        prev_features = [#'neg',
                         #'neu',
                         #'pos',
                         #'compound',
                         'sentence_cnt',
                         'word_cnt',
                         'punctuation_cnt',
                         'capital_cnt',
                         'upper_word_cnt',
                         'avg_word_cnt',
                         'avg_punc_cnt',
                         'avg_capital_cnt',
                         'avg_upper_cnt']

        # Vector assemble all features into one column called features.
        assemblerInputs = ['review_vector'] + numericCols + prev_features

        # Add pca to features if user wants.
        if pca:
            assemblerInputs += ['pcaFeatures']
            assemblerInputs.remove('review_vector')

        assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="unstandard_features")
        stages += [assembler]

        # Standardize features.
        scaler = StandardScaler(inputCol="unstandard_features", outputCol="features",
                                withStd=True, withMean=False)
        stages += [scaler]

        # Initialize the pipeline with the stages that were set.
        pipeline = Pipeline(stages=stages)

        # Fit the pipeline to training documents.
        pipelineFit = pipeline.fit(self.df)
        self.df = pipelineFit.transform(self.df)

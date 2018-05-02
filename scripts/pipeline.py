from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, udf, coalesce
import utils

##############
# Pipeline
##############

class Pipeline():

    def __init__(self, pyspark_df,spark_session):
        self.df = pyspark_df
        self.session = spark_session

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

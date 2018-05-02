############
# Utils.py
############

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, udf, coalesce
import re

@udf('double')
def ith(v, i):
    '''
    Helper function used to get an index of a column that has vectors
    as its values.

    Input:
    -------
    v : int column
    i : int index

    Output:
    -------
    float : Value of what's in v[i]
    '''
    try:
        return float(v[i])
    except ValueError:
        return None

@udf('double')
def sentence_count(review):
    '''
    User defined function to get number of sentences in a single review.

    Input:
    -------
    str : review

    Output:
    -------
    float : number of sentences in review
    '''
    s = review
    replacements = ['?','!']
    for char in replacements:
        s = s.replace(char,'.')
    return float(len(s.split('.')))

@udf('double')
def word_count(review):
    '''
    User defined function to get number of words in a single review.

    Input:
    -------
    str : review

    Output:
    -------
    float : number of words in review
    '''
    s = review
    return float(len(s.split(' ')))

@udf('double')
def count_punctuation(review):
    '''
    User defined function to get number of punctuation marks
    in a single review.

    Input:
    -------
    str : review
    list : punctuations you want to count

    Output:
    -------
    float : number of puncuations in review
    '''
    s = review
    count = 0
    punctuations = ['!']
    for char in punctuations:
        count += s.count(char)
    return float(count)


@udf('double')
def count_capital(review):
    '''
    User defined function to get number of capital letters in the review.

    Input:
    -------
    str : review

    Output:
    -------
    float : number of uppercase letters in review
    '''
    return float(len(re.findall(r'[A-Z]',review)))

@udf('double')
def all_caps(review):
    '''
    Takes a string of text and counts the number of ALL UPPERCASE words.

    Input:
    -------
    str : review

    Output:
    -------
    float : number of words that are in all uppsercase
    '''
    return float((len(re.findall('\s([A-Z][A-Z]+)', review))))

@udf('int')
def overall_transform(overall):
    '''
    User defined function to return whether a rating is greater
    than or equal to the threshold.

    Input:
    -------
    int : overall
    int : threshold

    Output:
    -------
    int : 1 if overall >= threhold, else 0
    '''
    if overall >= 4.0:
        return 1
    else:
        return 0

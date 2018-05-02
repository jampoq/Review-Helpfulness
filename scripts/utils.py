############
# Utils.py
############

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, udf, coalesce

@udf('double')
def ith(v, i):
    '''
    Helper function used to get an index of a column that has vectors
    as its values.

    Input:
    -------
    v : column
    i : index

    Output:
    -------
    Value of what's in v[i]
    '''
    try:
        return float(v[i])
    except ValueError:
        return None

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 23:39:03 2018

@author: jtotten
"""

import os

# os.chdir("C:/Users/JTOTTEN/Desktop/machinelearning_poc")

### Set up SparkSession
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SQLContext
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, coo_matrix, vstack
from pyspark.sql.functions import col
from pyspark.mllib.regression import LabeledPoint
import scipy.sparse
from pyspark.ml.linalg import Vectors, _convert_to_vector, VectorUDT
from pyspark.sql.functions import udf, col
from pyspark.ml.feature import OneHotEncoder, StringIndexer


sc = SparkContext(appName = "BuildProductRecommendations")
sql_sc = SQLContext(sc)

spark = SparkSession \
    .builder \
    .appName("Python Spark Create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate
    
#create a pd data frame
train_df = pd.read_csv('C:/Users/JTOTTEN/Desktop/machinelearning_poc/train_data/train_set_10_2.csv')


## enumerate pd data frame
prodSet = set(train_df.product_id)
catSet = set(train_df.prod_cat)
atrSet = set(train_df.prod_attr)

def enumerateSet(s):
    d = {}
    i = 0
    for elem in s:
        d.update({elem: i})
        i+=1
    return d

pEnum = enumerateSet(prodSet)
cEnum = enumerateSet(catSet)
aEnum = enumerateSet(atrSet)

p = [pEnum[x] for x in list(train_df.product_id)]
c = [cEnum[x] for x in list(train_df.prod_cat)]
a = [aEnum[x] for x in list(train_df.prod_attr)]

# create a column with the enumerated data
train_df['pEnum'] = [pEnum[x] for x in list(train_df.product_id)]
train_df['cEnum'] = [cEnum[x] for x in list(train_df.prod_cat)]
train_df['aEnum'] = [aEnum[x] for x in list(train_df.prod_attr)]

## create a new data frame
train_df2 = train_df[['customer_id', 'pEnum', 'cEnum', 'aEnum', 'buy']]

# jw function
def dense_to_sparse(vector):
    return _convert_to_vector(scipy.sparse.csc_matrix(vector.toArray()).T)


######
import findspark
findspark.init()
#findspark.init('C:/Users/JTOTTEN/Desktop/opt/spark')

# create RDD from enumerated pd data frame
sparktrain_df = sql_sc.createDataFrame(train_df2)
sparktrain_df.show(5)
sparktrain_df.printSchema()
sparktrain_df.dtypes

#from pyspark.sql.functions import struct
#sparktrain_df.withColumn("fTuple",struct(sparktrain_df.customer_id,))


######
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

def oneHot(inputData, inputCol, outputCol, middleCol):
    stringIndexer = StringIndexer(inputCol=inputCol, outputCol = middleCol)
    ohc = OneHotEncoder(inputCol=middleCol, outputCol = outputCol)

    model = stringIndexer.fit(inputData)
    indexed = model.transform(inputData)

    encoded = ohc.transform(indexed)
    return encoded

ohP = oneHot(sparktrain_df, "pEnum", "pOH", "x")
ohC = oneHot(ohP, "cEnum", "cOH", "y")
ohA = oneHot(ohC, "aEnum", "aOH", "z")
ohA.show()
assembler = VectorAssembler(inputCols=["pOH", "cOH", "aOH"], outputCol = "features")
output = assembler.transform(ohA)
output.show(5)


lp = (output.select(col("buy").alias("label"), (col("features")))).rdd.map(lambda row: (row.label, row.features))
lp.collect()























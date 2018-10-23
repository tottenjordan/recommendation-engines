# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 08:10:56 2018

@author: jtotten
"""

import os
os.chdir("C:/Users/JTOTTEN/Desktop/Data/machinelearning_poc")

import numpy as np
import scipy
from scipy.sparse import csr_matrix
import pandas as pd
import boto3
import re
import numpy as np
import math
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import nltk
#from nltks.corpus import stopwords
nltk.download("stopwords")

################  Data ########################################################

trainData = pd.read_csv('bestbuy_datav2.csv')
trainData.dtypes

trainData.head(5)
trainData.rename(columns={'transaction': 'purchase_count'}, inplace=True)
trainData.rename(columns={'Unnamed: 0': 'transaction_id'}, inplace=True)
# trainData.head(5)
# trainData.dtypes
trainData.loc[:,('product_id')] = trainData.product_id.astype(np.int64)
trainData.loc[:,('purchase_count')] = trainData.purchase_count.astype(np.float64)
trainData = trainData[np.isfinite(trainData['customer_id'])]
trainData = trainData[np.isfinite(trainData['product_id'])]
trainData.loc[:,('customer_id')] = trainData.customer_id.astype(np.int64)
trainData['prod_attr'].fillna('MISC', inplace=True)
# trainData.to_csv('bestbuy_datav4.csv', index=False)
# trainData.dtypes
# bestbuy_sample = trainData.head(10000)
# bestbuy_sample.to_csv('bestbuy_sample.csv')

# test row vs data frame
# print(type(trainData.product_id.iloc[0]))
# print(type(trainData.iloc[0].product_id))

# items dataframe
item_df = trainData[['product_id', 'prod_name', 'prod_cat', 'prod_attr']]
items_df = item_df.drop_duplicates(["product_id"])
# item_df['prod_attr'].fillna('MISC', inplace=True)
items_df.head(10)
items_df.dtypes
# rest index
items_df = items_df.reset_index(drop=True)
items_df.index = np.arange(1, len(items_df) + 1)

# purchases dataframe
purchases_df = trainData[['customer_id', 'product_id', 'purchase_count']]
purchases_df.dtypes

# test row vs data frame
# print(type(purchases_df.product_id.iloc[0]))
# print(type(trainData.iloc[0].product_id))


# COLD START - remove customers with less than 5 past purchases
buyer_purchases_count_df = purchases_df.groupby(['customer_id','product_id']).size().groupby('customer_id').size()
print('# buyers: %d' % len(buyer_purchases_count_df))
model_buyers_df = buyer_purchases_count_df[buyer_purchases_count_df >= 5].reset_index()[['customer_id']]
print('# buyers with at least 5 purchases: %d' % len(model_buyers_df))
model_buyers_df.head(5)

print('# of purchases: %d' % len(purchases_df))
purchases_from_model_buyers_df = purchases_df.merge(model_buyers_df,
            how = 'right',
            left_on = 'customer_id',
            right_on = 'customer_id')
print('# of purchases from users with at least 5 interactions: %d' % len(purchases_from_model_buyers_df))
purchases_from_model_buyers_df.head(5)

###############################################################################
################  Evaluation  #################################################
# cross-validation technique - holdout
# hold out XX% of the data for evaluation
# Take more than 15 minutes to compute

purchases_full_df = purchases_from_model_buyers_df.groupby(['customer_id', 'product_id'])['purchase_count'].sum().reset_index() 
print('# of unique customer/purchase interactions: %d' % len(purchases_full_df))
# purchases_full_df.head(5)
# purchases_full_df.dtypes


##### Hold Out - Create Test and Train Data Sets
purchases_train_df, purchases_test_df = train_test_split(purchases_full_df, 
                                    stratify=purchases_full_df['customer_id'], 
                                    test_size=0.30, 
                                    random_state=42)

print('# purchases on Train set: %d' % len(purchases_train_df))
print('# of purchases on Test set: %d' % len(purchases_test_df))
# train set = 2,654,445
# test set = 1,137,620

# Write Test and Train to CSV for storage
# purchases_train_df.to_csv("purchases_train_aug_2.csv")
# purchases_test_df.to_csv("purchases_test_aug_2.csv")

##################### purchases & Test split #######################################################

purchases_train_df = pd.read_csv('purchases_train_aug_2.csv')
purchases_test_df = pd.read_csv('purchases_test_aug_2.csv')
df_to_merge = trainData[['transaction_id', 'customer_id', 'product_id', 'prod_name', 'prod_cat', 'prod_attr']]
df_to_merge2 = trainData[['transaction_id', 'customer_id', 'product_id', 'prod_name', 'prod_cat', 'prod_attr']]

purchases_train_df.rename(columns={'Unnamed: 0': 'transaction_id'}, inplace=True)
purchases_test_df.rename(columns={'Unnamed: 0': 'transaction_id'}, inplace=True)

train_transactions = purchases_train_df[['transaction_id', 'purchase_count']]
test_transactions = purchases_test_df[['transaction_id', 'purchase_count']]

df_train = pd.merge(train_transactions, df_to_merge, on='transaction_id', how="left")
df_test = pd.merge(test_transactions, df_to_merge2, on='transaction_id', how="left")

df_train.dtypes
df_train.loc[:,('product_id')] = df_train.product_id.astype(np.float32)
df_train.loc[:,('customer_id')] = df_train.customer_id.astype(np.float32)
df_train.loc[:,('purchase_count')] = df_train.purchase_count.astype(np.float32)
df_train.dtypes

df_test.dtypes
df_test.loc[:,('product_id')] = df_test.product_id.astype(np.float32)
df_test.loc[:,('customer_id')] = df_test.customer_id.astype(np.float32)
df_test.loc[:,('purchase_count')] = df_test.purchase_count.astype(np.float32)
df_test.dtypes

df_train.to_csv("purchases_train_aug_18.csv", index=False)
df_test.to_csv("purchases_test_aug_18.csv", index=False)

#####################################################################################################



purchases_full_indexed_df = purchases_full_df.set_index('customer_id')
purchases_train_indexed_df = purchases_train_df.set_index('customer_id')
purchases_test_indexed_df = purchases_test_df.set_index('customer_id')


def get_products_purchased(person_id, purchases_df):
    # Get the buyer's data and merge 
    purchased_products = purchases_df.loc[person_id]['product_id']
    return set(purchased_products if type(purchased_products) == pd.Series else [purchased_products])

# test get_products_purchased
get_products_purchased(6, purchases_full_indexed_df)


# Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_PURCHASED_PRODUCTS = 100

class ModelEvaluator:
    
    
    def get_not_purchased_products_sample(self, person_id, sample_size, seed=42):
        purchased_products = get_products_purchased(person_id, purchases_full_indexed_df)
        all_products = set(items_df['product_id'])
        non_purchased_products = all_products - purchased_products
        
        random.seed(seed)
        non_purchased_products_sample = random.sample(non_purchased_products, sample_size)
        return set(non_purchased_products_sample)
    
    def _verify_hit_top_n(self, item_id, recommended_products, topn):
        try:
            index = next(i for i, c in enumerate(recommended_products) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index
    
    def evaluate_model_for_customer(self, model, person_id):
        # Getting the items in test set
        purchased_values_testset = purchases_test_indexed_df.loc[person_id]
        if type(purchased_values_testset['product_id']) == pd.Series:
            customer_purchased_products_testset = set(purchased_values_testset['product_id'])
        else:
            customer_purchased_products_testset = set([int(purchased_values_testset['product_id'])])
        purchased_products_count_testset = len(customer_purchased_products_testset)
        
        # Getting a ranked recommendation list from a model for a given customer
        customer_recs_df = model.recommend_products(person_id,
                                                    products_to_ignore=get_products_purchased(person_id, 
                                                                                              purchases_train_indexed_df),
                                                                                              topn=10000000000)
        
        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the customer has purchased in test set
        for item_id in customer_purchased_products_testset:
            # Getting a random sample (100) products the user has not purchased
            # (to represent products that are assumed to be not relevant to the customer)
            non_purchased_products_sample = self.get_not_purchased_products_sample(person_id,
                                                                                   sample_size=EVAL_RANDOM_SAMPLE_NON_PURCHASED_PRODUCTS,
                                                                                   seed=item_id%(2**32))
            # Combining the current purchased item with the 100 random products
            products_to_filter_recs = non_purchased_products_sample.union(set([item_id]))
            
            # filtering only recommendations that are either the purchased products or from a random sample
            # of 100 non-purchased items
            valid_recs_df = customer_recs_df[customer_recs_df['product_id'].isin(products_to_filter_recs)]
            valid_recs = valid_recs_df['product_id'].values
            # Verifying if the current purchased product is among the Top-N recommended products
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10
            
        # Recall is the rate of the purchased products that are ranked among the Top-N recommended
        # products, when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(purchased_products_count_testset)
        recall_at_10 = hits_at_10_count / float(purchased_products_count_testset)
        
        customer_metrics = {'hits@5_count':hits_at_5_count,
                            'hits@10_count':hits_at_10_count,
                            'interacted_count':purchased_products_count_testset,
                            'recall@5':recall_at_5,
                            'recall@10':recall_at_10}
        return customer_metrics
    
    def evaluate_model(self, model):
        # print ('Running evaluation for customers')
        customerss_metrics = []
        for idx, person_id in enumerate(list(purchases_test_indexed_df.index.unique().values)):
             #if idx % 100 == 0 and idx >0:
                # print('%d customers processed' % idx)
            customer_metrics = self.evaluate_model_for_customer(model, person_id)
            customer_metrics['_person_id'] = person_id
            customerss_metrics.append(customer_metrics)
        print('%d customers processed' % idx)
        
        detailed_results_df = pd.DataFrame(customerss_metrics).sort_values('purchase_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['purchase_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['purchase_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()


##################### Popularity Model ########################################

product_popularity_df = purchases_full_df.groupby('product_id')['purchase_count'].sum().sort_values(ascending=False).reset_index()
product_popularity_df.head(10)
product_popularity_df.dtypes

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, products_df=None):
        self.popularity_df = popularity_df
        self.products_df = products_df
        
    def get_model_name(self):
        return self.MODEL_NAME
    
    def recommend_products(self, user_id, products_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular products that the customer hasn't seen yet
        recommendations_df = self.popularity_df[~self.popularity_df['product_id'].isin(products_to_ignore)] \
                                .sort_values('purchase_count', ascending = False).head(topn)
        
        if verbose:
            if self.products_df is None:
                raise Exception('"products_df" is required in verbose mode')
                
            recommendations_df = recommendations_df.merge(self.products_df, how = 'left',
                                                          left_on = 'product_id',
                                                          right_on = 'product_id')[['purchase_count', 'product_id', 'prod_name', 'prod_cat', 'prod_attr']]
            
        return recommendations_df

popularity_model = PopularityRecommender(product_popularity_df, items_df)


print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)

# Test

# popularityrecommender2 = PopularityRecommender(product_popularity_df, items_df)
# customer_recs_test_df = popularityrecommender2.recommend_products(6, products_to_ignore=get_products_purchased(6, purchases_train_indexed_df), topn=10000000000)

# ignore = get_products_purchased(6, purchases_train_indexed_df)
# rec_test = product_popularity_df[~product_popularity_df['product_id'].isin(ignore)].sort_values('purchase_count', ascending=False).head(10)

###############################################################################
##################### Content Based Filtering #################################
###############################################################################

# leverage description or attributes from items the user has purchased
# depends only on the user previous choices, avoids the COLD START problem
# fix items data
item_df['prod_attr'].fillna('MISC', inplace=True)

#Igrnoring stopwords (words with no semantics) from English 
stopwords_list = stopwords.words('english')

# Trains a model whose vectors size is 5000, composed by the main unigrams and 
# bigrams found in the corpus, ignoring stopwords

vectorizer = TfidfVectorizer(analyzer='word',
                             ngram_range=(1, 2),
                             min_df=0.003,
                             max_df=0.5,
                             max_features=5000,
                             stop_words=stopwords_list)

product_ids = items_df['product_id'].tolist()
tfidf_matrix = vectorizer.fit_transform(items_df['prod_name'] + "" + items_df['prod_attr'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix

# To model a customer, take all the item profiles the user purchased and average them. 
# the average is weighted by the purchases count strength, i.e., the items customer has 
# purchased the most will have a higher stregnth in the final user profile

def get_product_profile(product_id):
    idx = product_ids.index(product_id)
    product_profile = tfidf_matrix[idx:idx+1]
    return product_profile

# get_product_profile(3322043) #ok works

def get_product_profiles(ids):
    product_profiles_list = [get_product_profile(x) for x in ids]
    product_profiles = scipy.sparse.vstack(product_profiles_list)
    return product_profiles

def build_customers_profile(person_id, purchases_indexed_df):
    purchases_customer_df = purchases_indexed_df.loc[person_id]
    customer_product_profiles = get_product_profiles(purchases_customer_df['product_id'])
    
    customer_product_strengths = np.array(purchases_customer_df['purchase_count']).reshape(-1,1)
    # weighted average of product profiles by the interactions strength
    customer_product_strengths_weighted_avg = np.sum(customer_product_profiles.multiply(customer_product_strengths), axis=0) / np.sum(customer_product_strengths)
    customer_profile_norm = sklearn.preprocessing.normalize(customer_product_strengths_weighted_avg)
    return customer_profile_norm

# build_customers_profile(3322043, purchases_full_df)

def build_customers_profiles():
    purchases_indexed_df = purchases_full_df[purchases_full_df['product_id'].isin(items_df['product_id'])].set_index('customer_id')
    
    customer_profiles = {}
    for person_id in purchases_indexed_df.index.unique():
        customer_profiles[person_id] = build_customers_profile(person_id, purchases_indexed_df)
    return customer_profiles

customer_profiles = build_customers_profiles()
len(customer_profiles)


# Test a customer profile
# The value in each position represents how relevant a token is
myprofile = customer_profiles[6]
print(myprofile.shape)
pd.DataFrame(sorted(zip(tfidf_feature_names, 
                        customer_profiles[6].flatten().tolist()), key=lambda x: -x[1])[:20], 
             columns=['token', 'relevance'])

# Content Based Class
  
class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, products_df=None):
        self.product_ids = product_ids
        self.products_df = products_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_products_to_customer_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(customer_profiles[person_id], tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_products = sorted([(product_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_products
        
    def recommend_products(self, user_id, products_to_ignore=[], topn=10, verbose=False):
        similar_products = self._get_similar_products_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_products_filtered = list(filter(lambda x: x[0] not in products_to_ignore, similar_products))
        
        recommendations_df = pd.DataFrame(similar_products_filtered, columns=['product_id', 'rec_strength']) \
                                    .head(topn)

        if verbose:
            if self.products_df is None:
                raise Exception('"products_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.products_df, how = 'left', 
                                                          left_on = 'product_id', 
                                                          right_on = 'product_id')[['rec_strength', 'product_id', 'prod_name', 'prod_cat', 'prod_attr']]


        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(items_df)

print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)


###############################################################################
############################### Collaborative Filtering #######################
###############################################################################

########################### Matrix Facotrization ###############################

#Creating a sparse pivot table with users in rows and items in columns
customers_products_pivot_matrix_df = purchases_train_df.pivot(index='customer_id', 
                                                          columns='product_id', 
                                                          values='purchase_count').fillna(0)

customers_products_pivot_matrix_df.head(10)

customers_products_pivot_matrix = customers_products_pivot_matrix_df.as_matrix()
customers_products_pivot_matrix[:10]

customers_ids = list(customers_products_pivot_matrix_df.index)
customers_ids[:10]


# Determine the number of factors to factor the user-item matrix.
# NUMBER_OF_FACTORS_MF = 10

#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(customers_products_pivot_matrix, k = NUMBER_OF_FACTORS_MF)

U.shape

Vt.shape

sigma = np.diag(sigma)
sigma.shape

# After facotrization, reconstruct the original matrix by multiplying its factors
# THIS MATRIX IS NOT SPARSE
# This is generated predictions for items the customer has not yet purchased
# This will be used for recommendations

all_customer_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_customer_predicted_ratings

#Converting the reconstructed matrix back to dataframe
cf_preds_df = pd.DataFrame(all_customer_predicted_ratings, columns = customers_products_pivot_matrix_df.columns, index=customers_ids).transpose()
cf_preds_df.head(10)

len(cf_preds_df.columns)

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, products_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.products_df = products_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, products_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_customer_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'rec_strength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_customer_predictions[~sorted_customer_predictions['product_id'].isin(products_to_ignore)] \
                               .sort_values('rec_strength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.products_df is None:
                raise Exception('"products_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.products_df, how = 'left', 
                                                          left_on = 'product_id', 
                                                          right_on = 'product_id')[['rec_strength', 'product_id', 'prod_name', 'prod_cat', 'prod_attr']]


        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, items_df)

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)


















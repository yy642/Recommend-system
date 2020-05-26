from __future__ import print_function, division
from peewee import *
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import pickle
from models import *

db = MySQLDatabase("spider", host="127.0.0.1", port=3306, user="root", password="")

"""
(1) select out authors who had more than 3 ratings
"""
required_reviews = 3

i_author = 0
authors = {}
query_authors=Review.select(Review.author_id, fn.count(Review.product_id).alias('count') ).group_by(Review.author_id)
for review in query_authors:
    if review.count >= required_reviews:
        authors[review.author_id] = i_author
        i_author += 1

"""
(2) select out products with more than 3 ratings
"""
i_product = 0
products = {}
query_products=Review.select(Review.product_id, fn.count(Review.author_id).alias('count') ).group_by(Review.product_id)
for review in query_products:
    if review.count >= required_reviews:
        products[review.product_id] = i_product
        i_product += 1

print("total number of authors who gave more than" , str(required_reviews), "reviews = ", len(authors))
print("total number of products who gave more than", str(required_reviews), "reviews = ", len(products))
with open('Authors.json', 'w') as fp:
    json.dump(authors, fp)
with open('Products.json', 'w') as fp:
    json.dump(products, fp)
authors_set = set(authors.keys())
products_set = set(products.keys())

"""
(3) write function to split training and testing set based on review time.
"""
#author2product = defaultdict(list)
#product2author = defaultdict(list)
#author_product_rating = defaultdict(int)
author_product_rating = np.ones([len(authors), len(products)]) * (-1)


query = (Review.select(Review.author_id, Review.product_id, Review.rating, Review.time).order_by(Review.time))

for review in query:
    if review.author_id in authors_set and review.product_id in products_set:
        i = authors[review.author_id]
        j = products[review.product_id]
        #author_product_rating[(authors[review.author_id], products[review.product_id])] = review.rating
        author_product_rating[i, j] = review.rating


total_valid_reviews = np.sum(author_product_rating != -1)
training_ratio = 0.8

training_set = np.ones([len(authors), len(products)]) * (-1)
testing_set  = np.ones([len(authors), len(products)]) * (-1)
c = 0.
for review in query:
    if review.author_id in authors_set and review.product_id in products_set:
        i = authors[review.author_id]
        j = products[review.product_id]
        c += 1.
        if c / total_valid_reviews < training_ratio:
            training_set[i, j] = review.rating
        else:
            testing_set[i, j] = review.rating


print("valid review=",c, ",",c / len(authors) / len(products))

print("training", np.sum(training_set != -1) /total_valid_reviews )
print("testing", np.sum(testing_set != -1) /total_valid_reviews)
np.save("training_mat.npy", training_set)
np.save("testing_mat.npy", testing_set)

"""

with open('Author2product.json', 'w') as fp:
    json.dump(author2product, fp)
with open('Product2author.json', 'w') as fp:
    json.dump(product2author, fp)
with open('Author_product_rating.json', 'w') as fp:
    json.dump(author_product_rating, fp)
"""

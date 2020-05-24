from __future__ import print_function, division


from peewee import *
import pandas as pd

from models import *

db = MySQLDatabase("spider", host="127.0.0.1", port=3306, user="root", password="")

authors_dict = dict()
i_author = 0
products_dict = dict()
i_product = 0
for review in Review.select():
    print(review.author_id, review.product_id, review.rating)
    if review.author_id not in authors_dict.keys():
        authors_dict[review.author_id] = i_author
        i_author += 1
    if review.product_id not in products_dict.keys():
        products_dict[review.product_id] = i_product
        i_product += 1

print(len(authors_dict ))
print(len(products_dict ))
for i in authors_dict.keys():
    for j in products_dict.keys():
        ratings[i][j] = review
ratings = np.ones([len(authors_dict), len(products_dict)]) * (-1)
for review in Review.select():
    i = authors_dict[review.author_id]
    j = products_dict[review.product_id]
    ratings[i][j] = review.rating

print(np.mean(rating[rating != -1]))
#review_mat = np.zeros([])
#for p in Product.select():
#    print(p.product_id)

"""

from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future



# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('../large_files/movielens-20m-dataset/rating.csv')



# note:
# user ids are ordered sequentially from 1..138493
# with no missing numbers
# movie ids are integers from 1..131262
# NOT all movie ids appear
# there are only 26744 movie ids
# write code to check it yourself!


# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
  movie2idx[movie_id] = count
  count += 1

# add them to the data frame
# takes awhile
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

df = df.drop(columns=['timestamp'])

df.to_csv('../large_files/movielens-20m-dataset/edited_rating.csv', index=False)
"""
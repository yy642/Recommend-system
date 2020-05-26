import json
import numpy as np

with open('Authors.json') as fp:
    authors = json.load(fp)
print(authors)
print(len(authors))

with open('Products.json') as fp:
    products = json.load(fp)
print(products)
print(len(products))

X=np.load("training_mat.npy")
print(X)

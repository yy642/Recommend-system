import re
import ast
from urllib import parse
from datetime import datetime
from peewee import *
import requests
from get_review import *

from scrapy import Selector
from models import *
db = MySQLDatabase("spider", host="127.0.0.1", port=3306, user="root", password="")


if __name__ == "__main__":

    db.create_tables([Review])
    db.create_tables([Product])
    
    """
    get and save all products
    """
    products = requests.get("https://www.everlane.com/api/v3/collections/womens-all").json()['products']
    print("total # of products = ", len(products))
    for product in products:
        p = Product()
        setattr(p, 'product_id', int(product['id']))
        setattr(p, 'link', product['permalink'])
        setattr(p, 'price', float(product['price']))
        p.save()


    """
    get and save review
    """
    index = 0
    for product in products:
        product_id = product['id']
        q = get_review_query(0, 1, product_id)
        preview = requests.get(q).json()
        total_reviews = preview['TotalResults']
        remaining_reviews = total_reviews
        print(index, "total reviews=",total_reviews)
        index += 1
        for i in range(total_reviews // 100 + 1):
            limit = min(100, remaining_reviews)
            q = get_review_query(i * 100, limit, product_id)
            remaining_reviews -= limit
            tmp_reviews = requests.get(q).json()['Results']
            for review in tmp_reviews:
                author_id = int(review['AuthorId'])
                product_id = int(review['ProductId'])
                rating = int(review['Rating'])
                review_text = review['ReviewText']
                review_title = review['Title']
                p = Review()
                setattr(p, 'author_id', author_id)
                setattr(p, 'product_id', product_id)
                setattr(p, 'rating', rating)
                setattr(p, 'title', review_title)
                setattr(p, 'review', review_text)
                p.save()




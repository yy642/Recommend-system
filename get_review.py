pre = "https://www.everlane.com/api/v2/reviews/filter?reviews%5Bdata%5D%5BInclude%5D=Products&reviews%5Bdata%5D%5BStats%5D=Reviews&reviews%5Bdata%5D%5BLimit%5D="
mid = "&reviews%5Bdata%5D%5BOffset%5D="
tail = "&reviews%5Bfilters%5D%5BFilter%5D%5B%5D=ProductId%"
def get_review_query(offset, limit, product_id):
    query = pre + str(limit) + mid + str(offset) + tail + '3A' + str(product_id)
    return query
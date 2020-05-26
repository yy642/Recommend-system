from peewee import *

db = MySQLDatabase("spider", host="127.0.0.1", port=3306, user="root", password="")

class BaseModel(Model):
    class Meta:
        database = db

"""
char类型， 要设置最大长度
对于无法确定最大长度的字段，可以设置为Text
设计表的时候 采集到的数据要尽量先做格式化处理
default和null=True
"""



class Product(BaseModel):
    product_id = IntegerField(primary_key=True)
    link = TextField(default="")
    price = FloatField(default=0.0)


#class Customer(BaseModel):
#    author_id = IntegerField()

class Review(BaseModel):
    product_id = IntegerField(default=-1)
    author_id = IntegerField(default=-1)
    rating = IntegerField(default=-1)
    title = TextField(default="")
    review = TextField(default="")
    time = DateTimeField()
    class Meta:
        indexes = (
            (('product_id', 'author_id', 'time'), True),
        )

#if __name__ == "__main__":
#    db.create_tables([Review, Product, Customer])

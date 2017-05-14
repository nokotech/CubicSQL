try:
    from pymongo.connection import Connection
except ImportError as e:
    from pymongo import MongoClient as Connection

class MongoClass:

    ########################################################
    # コンストラクタ
    def __init__(self):
        connect = Connection('localhost', 27017)
        self.db = connect.test
        print("db name is = " + self.db.name)

    ########################################################
    def save(self, obj):
        collect = self.db.image
        #collect.save({'x':10})
        collect.save(obj)
        print("find_one = ")
        print(collect.find_one())

    ########################################################
    def find(self, query):
        collect = self.db.image
        print("find = ")
        for data in collect.find():
            print(data)
        print("find_query = ")
        #for data in collect.find({'x':10}):
        for data in collect.find(query):
            print(data)

    ########################################################
    def getDb(self):
        return self.db

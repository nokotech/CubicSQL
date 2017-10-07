from mongo import MongoClass

if __name__ == '__main__':
    m = MongoClass()
    m.save()
    m.find()
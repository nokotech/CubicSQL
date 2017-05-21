#!/usr/bin/env python

import sys
from mongo import *

if __name__ == '__main__':
    print(mongo.MongoClass)
    m = mongo.MongoClass()
    m.save()
    m.find()
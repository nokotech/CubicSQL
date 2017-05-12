#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
try:
    from pymongo.connection import Connection
except ImportError as e:
    from pymongo import MongoClient as Connection

if __name__ == '__main__':
    #connection
    connect = Connection('localhost', 27017)
    db = connect.test

    print("db name is = ")
    print(db.name)

    collect = db.foo
    collect.save({'x':10})
    collect.save({'x':8})
    collect.save({'x':11})

    print("find_one = ")
    print(collect.find_one())

    print("find = ")
    for data in collect.find():
        print(data)

    print("find_query = ")
    for data in collect.find({'x':10}):
        print(data)

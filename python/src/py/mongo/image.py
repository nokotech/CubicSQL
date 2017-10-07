import os
import sys
import mongo
import cv2
import numpy as np
import json

class ImageMongo:
    
    ########################################################
    # コンストラクタ
    def __init__(self):
        self.db = mongo.MongoClass()

    ########################################################
    def path(self, original_path):
        imageArray = []
        files = os.listdir(original_path)
        files_dir = [f for f in files if os.path.isdir(os.path.join(original_path, f))]
        print(files_dir)
        index = 0
        for n, dir in enumerate(files_dir):
            dir_path = original_path + '/' + dir
            files = os.listdir(dir_path)
            for i, file in  enumerate(files):
                #print(i, dir, file)
                imageArray.append(dir + "/" + file)
        return imageArray

    ########################################################
    def insert(self, path):
        pic = cv2.imread(path)
        pic = cv2.resize(pic, (28, 28))
        pic = pic.flatten().astype(np.float32)/255.0
        pic = json.dumps(pic, ensure_ascii=False)
        doc = {"name": path, "data": pic}
        print(doc)
        self.db.save(doc)

    ########################################################
    def find(self):
        self.db.find()

if __name__ == '__main__':
    PATH = "../../../temp/img"
    iMon = ImageMongo()
    imageArray = iMon.path(PATH)
    for i, a in enumerate(imageArray):
        iMon.insert(PATH + "/" + a)
    #iMon.find()

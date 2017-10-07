# coding: utf-8

import cv2
import sys
import os

# 定数
FACE_COLOR = (255, 0, 0)
IMAGE_SIZE = 112

#######################################################
## 
#######################################################
def faceCut(path, dir, index):

    # cascadeファイルをロードする
    cascades_dir = "../python/haarcascades/haarcascade_frontalface_alt2.xml"
    cascade_f = cv2.CascadeClassifier(cascades_dir)

    # 画像を読み込む
    img = cv2.imread('../image/original' + '/' + dir + '/' + path)

    # グレイスケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 検出して四角で囲む
    face = cascade_f.detectMultiScale(gray)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), FACE_COLOR, 2)

    # 切り抜き出来なかったらreturn
    if len(face) == 0 : return "---"

    # 切り抜き
    face = img[y:y+h, x:x+h]
    face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))

    #認識結果の保存
    ext = os.path.splitext(path)
    cv2.imwrite('../image/cut/' + dir + str(index) + ext, face)

    return ('../image/cut/' + dir + str(index) + ext)

#######################################################
## 
#######################################################
def faceMove(path, dir, index):

    # 画像を読み込む
    face = cv2.imread('../image/original' + '/' + dir + '/' + path)

     #認識結果の保存
    ext = os.path.splitext(path)
    cv2.imwrite('../image/cut/' + dir + str(index) + ext, face)

    return ('../image/cut/' + dir + str(index) + ext)


#######################################################
## 
#######################################################
if __name__ == '__main__':

    original_path = '../image/original'

    # テストデータ書き出し用
    train = open('../data/train.txt', 'w')
    test = open('../data/test.txt', 'w')

    # パスの抜き出し
    files = os.listdir(original_path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(original_path, f))]
    print(files_dir)

    index = 0
    for n, dir in enumerate(files_dir):
        print('---------------------------')
        dir_path = original_path + '/' + dir
        files = os.listdir(dir_path)
        for i, file in  enumerate(files):
            print(i, dir, file)
            file_path = dir_path + '/' + file
            #txt = faceCut(file, dir, i)
            txt = faceMove(file, dir, i)

            # 画像が抽出できなかったら破棄
            if txt == "---":
                os.rename(
                    '../image/original' + '/' + dir + '/' + file,
                    '../image/original' + '/' + 'discard_' + file
                )
                continue

            # テキストに書き出し
            if i <= 0:
                test.write(txt + ' ' + str(n) + '\n')
            else:
                train.write(txt + ' ' + str(n) + '\n')

    train.close()
    test.close()

    print('---------------------------')
    print('complete!')

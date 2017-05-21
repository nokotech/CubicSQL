import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

NUM_CLASSES = 3

"""
予測モデルを作成する関数
    @param images_placeholder - 画像のplaceholder
    @param keep_prob - dropout率のplace_holder
    @return y_conv - 各クラスの確率(のようなもの)
"""
def inference(images_placeholder, keep_prob):
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    # 畳み込み層の作成
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # プーリング層の作成
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 入力を28x28x3に変形
    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 3])
    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # 各ラベルの確率のようなものを返す
    return y_conv

"""
lossを計算する関数
    @param logits - ロジットのtensor, float - [batch_size, NUM_CLASSES]
    @param labels -  ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    @return cross_entropy: 交差エントロピーのtensor, float
"""
def loss(logits, labels):
    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    # TensorBoardで表示するよう指定
    # tf.scalar_summary("cross_entropy", cross_entropy) #~1.0
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

"""
訓練のOpを定義する関数
    @param loss - 損失のtensor, loss()の結果
    @param learning_rate - 学習係数
    @return train_step - 訓練のOp
"""
def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

"""
正解率(accuracy)を計算する関数
    @param logits - inference()の結果
    @param labels - ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    @return accuracy - 正解率(float)
"""
def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # tf.scalar_summary("accuracy", accuracy) #~1.0
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

"""
ファイルを展開する関数
    @param path - ファイルパス
    @return image - 画像
    @return label - ラベル
"""
def getfile(path):
    # ファイルを開く
    f = open('../data/training.txt', 'r')
    # データを入れる配列
    image = []
    label = []
    for line in f:
        # 改行を除いてスペース区切りにする
        line = line.rstrip()
        l = line.split()
        # パス
        p = l[0]
        # データを読み込んで28x28に縮小
        img = cv2.imread(p)
        img = cv2.resize(img, (28, 28))
        # 一列にした後、0-1のfloat値にする
        image.append(img.flatten().astype(np.float32)/255.0)
        # ラベルを1-of-k方式で用意する
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        label.append(tmp)
    # numpy形式に変換
    image = np.asarray(image)
    label = np.asarray(label)
    f.close()
    print(path + str(len(image)) + '枚ファイルを取得しました。')
    # print(image)
    # print(label)
    return image, label

"""
メイン関数
"""
if __name__ == '__main__':

    print('------------------------------------')
    train_image, train_label = getfile('../data/training.txt')
    test_image, test_label = getfile('../data/test.txt')
    print('------------------------------------')
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder("float", shape=(None, 28*28*3))
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        keep_prob = tf.placeholder("float")
        # inference → loss → training
        logits = inference(images_placeholder, keep_prob)
        loss_value = loss(logits, labels_placeholder)
        train_op = training(loss_value, 1e-4)
        acc = accuracy(logits, labels_placeholder)
        # Session
        saver = tf.train.Saver()
        sess = tf.Session()
        # sess.run(tf.initialize_all_variables()) # ~1.0
        sess.run(tf.global_variables_initializer())
        # TensorBoardで表示する値の設定
        # summary_op = tf.merge_all_summaries() # ~1.0
        summary_op =  tf.summary.merge_all()
        #summary_writer = tf.train.SummaryWriter('image/train/', sess.graph_def) # ~1.0
        summary_writer = tf.summary.FileWriter('../image/train/', sess.graph)
        # 訓練の実行
        for step in range(100):
            for i in range(1):
                batch = 10*i
                sess.run(train_op, feed_dict={images_placeholder: train_image[batch:batch+10], labels_placeholder: train_label[batch:batch+10], keep_prob: 0.5})
                train_accuracy = sess.run(acc, feed_dict={images_placeholder: train_image, labels_placeholder: train_label, keep_prob: 1.0})
                print("step %d, training accuracy %g"%(step, train_accuracy))
                summary_str = sess.run(summary_op, feed_dict={images_placeholder: train_image, labels_placeholder: train_label, keep_prob: 1.0})
                summary_writer.add_summary(summary_str, step)

    # 訓練が終了したらテストデータに対する精度を表示
    print("test accuracy %g"%sess.run(acc, feed_dict={images_placeholder: test_image, labels_placeholder: test_label, keep_prob: 1.0}))
    # 最終的なモデルを保存
    save_path = saver.save(sess, "../model/model.ckpt")
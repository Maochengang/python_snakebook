# 导入数据

import mne
import matplotlib.pyplot as pl
import numpy as np

# Mention the file path to the dataset
# filename根据自己的来
filename = r'C:\Users\mcg\Desktop\大三上\脑电\class 2\4-class MI\BCICIV_2a_gdf\A01T.gdf'

raw = mne.io.read_raw_gdf(filename)

print(raw.info)
print(raw.ch_names)

# 小波包分解
import pywt  # 导入小波变化库


def wpd(X):  # 宏定义wpd（x）函数，实例化的小波包对象
    coeffs = pywt.WaveletPacket(X, 'db4', mode='symmetric', maxlevel=5)  # 使用 'db4' 小波将信号分解到第 5 级
    return coeffs


def feature_bands(x):
    Bands = np.empty((8, x.shape[0], x.shape[1], 30))  # 8个频带系数从4-32Hz的范围内选择

    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
            pos = []
            C = wpd(x[i, ii, :])
            pos = np.append(pos, [node.path for node in C.get_level(5, 'natural')])  # 按自然顺序得到特定层上的所有节点
            # 把每一个epoch的每一通道（or每一通道的每一epoch）拿去做小波包分解，分解成一颗五层的二叉树

            for b in range(1, 9):
                Bands[b - 1, i, ii, :] = C[pos[b]].data

    return Bands


wpd_data = feature_bands(data)  # 将band的数据储存在data中

from mne.decoding import CSP  # 利用CSP提取特征，建立线性分类器
from sklearn import preprocessing  # python语言的机器学习工具包
from sklearn.preprocessing import OneHotEncoder  # 分类编码变量，将每一个类可能取值的特征变换为二进制特征向量
from tensorflow.keras.models import Sequential  # 序贯模型  #keras与tf版本不兼容，改为tensorflow.keras.XX
from tensorflow.keras.layers import Dense  # 本质是由一个特征空间线性变换到另一个特征空间，dense层的目的是将前面提取的特征，在dense经过非线性变化
from tensorflow.keras.layers import Dropout  # dropout 丢弃层，生成一个更瘦的神经网络
from tensorflow.keras import regularizers  # 正则化器允许在优化过程中对层的参数或层的激活情况进行惩罚，网络优化的损失函数也包括这些惩罚项
from sklearn.model_selection import ShuffleSplit  # 用于将样本集合随机“打散”后划分为训练集、测试集

# OneHotEncoding Labels
enc = OneHotEncoder()
X_out = enc.fit_transform(labels.reshape(-1, 1)).toarray()  # 对x进行编码，对于输入数组，依旧是把每一行当作一个样本，每一列当作一个特征

# Cross Validation Split 交叉验证拆分
cv = ShuffleSplit(n_splits=10, test_size=0.2,
                  random_state=0)  # n_splits:划分数据集的份数，类似于KFlod的折数，默认为10份test_size：测试集所占总样本的比例， random_state：随机数种子，使每次划分的数据集不

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc = []  # 精度
ka = []  # Kappa系数用于一致性检验，也可以用于衡量分类精度
prec = []  # 准确度
recall = []  # 召回率


#建立分类器模型
#Keras实现了很多层，包括core核心层，Convolution卷积层、Pooling池化层等非常丰富有趣的网络结构
def build_classifier(num_layers = 1):
    classifier = Sequential()     #序贯模型
    #第一层
    classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'relu', input_dim = 32,
                         kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
    classifier.add(Dropout(0.5))
    # 中间层
    for itr in range(num_layers):
        classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'relu',
                             kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
        classifier.add(Dropout(0.5))
    # 底层
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier


# StandardScaler是进行标准化/归一化的工具
# X_train——提取相关特征
# X_test——提取测试集相关特征

for train_idx, test_idx in cv.split(labels):
    Csp = [];
    ss = [];
    nn = []  # 创建空列表

    label_train, label_test = labels[train_idx], labels[test_idx]
    y_train, y_test = X_out[train_idx], X_out[test_idx]

    Csp = [CSP(n_components=4, reg=None, log=True, norm_trace=False) for _ in range(8)]  # 对所有频段系数分别应用CSP滤波器
    ss = preprocessing.StandardScaler()

    X_train = ss.fit_transform(
        np.concatenate(tuple(Csp[x].fit_transform(wpd_data[x, train_idx, :, :], label_train) for x in range(8)),
                       axis=-1))

    X_test = ss.transform(
        np.concatenate(tuple(Csp[x].transform(wpd_data[x, test_idx, :, :]) for x in range(8)), axis=-1))

    nn = build_classifier()

    nn.fit(X_train, y_train, batch_size=32, epochs=300)  # 训练模型

    y_pred = nn.predict(X_test)  # 预测
    pred = (y_pred == y_pred.max(axis=1)[:, None]).astype(int)

    acc.append(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)))  # 评分
    ka.append(cohen_kappa_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    prec.append(precision_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))
    recall.append(recall_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))

import pandas as pd

scores = {'Accuracy': acc, 'Kappa': ka, 'Precision': prec, 'Recall': recall}

Es = pd.DataFrame(scores)

avg = {'Accuracy': [np.mean(acc)], 'Kappa': [np.mean(ka)], 'Precision': [np.mean(prec)], 'Recall': [np.mean(recall)]}

Avg = pd.DataFrame(avg)

T = pd.concat([Es, Avg])

T.index = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'Avg']
T.index.rename('Fold', inplace=True)

print(T)
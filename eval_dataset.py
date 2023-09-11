import tensorflow as tf
import matplotlib.pyplot as pl
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix,f1_score
tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as pl
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
import utils
import numpy as np
dataset_root = os.path.join(os.getcwd(),"configs\\datasets\\RAVDESS")

# 矩阵可视化
def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
    '''
    plot_matrix函数参数包括：
    y_true样本的真实标签，为一向量
    y_pred样本的预测标签，为一向量，与真实标签长度相等
    labels_name样本在数据集中的标签名，如在示例中，样本的标签用0, 1, 2表示，则此处应为[0, 1, 2]
    title=None图片的标题
    thresh=0.8临界值，大于此值则图片上相应位置百分比为白色
    axis_labels=None最终图片中显示的标签名，如在示例中，样本标签用0, 1, 2表示分别表示失稳、稳定与潮流不收敛，我们最终图片中显示后者而非前者，则可令此参数为[‘unstable’, ‘stable’, ‘non-convergence’]
    '''
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar()  # 绘制图例
    # 图像标题
    if title is not None:
        pl.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    pl.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    pl.show()

def get_class_id(dir_name):
    return config.class_labels.index(dir_name)



if __name__ == '__main__':
    config = utils.parse_opt()
    model = models.load(config)
    first = True
    if not os.path.exists(os.path.join(config.feature_folder,"predict.csv")):
        for dir_name in os.listdir(dataset_root):
            class_id = get_class_id(dir_name)
            dir_path = os.path.join(dataset_root,dir_name)
            if first:
                of.get_data(config,dir_path,False,class_id,False)
                first = False
            else:
                of.get_data(config, dir_path, False, class_id, True)

    # X样本特征 Y样本标签
    X, Y = of.load_feature(config,False,show_label=True)
    # 对X的预测的置信率
    Y_predict = model.predict_proba(X)
    # 选出置信度最高的那个最为预测标签
    Y_predict_label = tf.argmax(Y_predict, axis=-1)

    # 计算每个情感的F1 scorce并画出矩阵
    '''
    print("模型标签与F1 score分数计算如下1:\n --------------------------------------------")
    print("\t".join(config.class_labels))
    f1_score = calculate_f1(Y, Y_predict_label)
    print("\t".join(str(min(max(round(num, 2), 0), 1)) for num in f1_score))
    plot_confuse(config, Y, Y_predict)
    '''
    print("模型标签与召回率计算如下:\n --------------------------------------------")
    print("召回率")
    print("\t".join(config.class_labels))
    recall_score = recall_score(Y, Y_predict_label,labels=[0, 1, 2, 3, 4, 5, 6, 7], average = None)
    print("\t".join(str(min(max(round(num, 2), 0), 1)) for num in recall_score))
    # plot_confuse(config, Y, Y_predict)
    C = confusion_matrix(Y,Y_predict_label)
    print("混淆矩阵")
    print(C)
    plot_matrix(Y, Y_predict_label, [0, 1, 2, 3, 4, 5, 6, 7], title='confusion_matrix_svc', axis_labels = None)

# TypeError: Singleton array <tf.Tensor: shape=(), dtype=int64, numpy=7> cannot be considered a valid collection.警示框出现，删除predict.csv即可。

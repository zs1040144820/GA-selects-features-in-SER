from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

data = make_blobs(n_samples = 1000, centers = 6, random_state = 8)
print(data)

X,y = data
import matplotlib.pyplot as plt


plt.scatter(X[:,0],X[:,1],c = y,cmap=plt.cm.spring,edgecolors="k")
plt.show()

#划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X,y ,random_state=0)
print('X_train:{},X_test:{}'.format(X_train.shape,X_test.shape))
clf = KNeighborsClassifier()#knn分类器实例化
clf.fit(X_train,y_train)#模型训练
print('测试集评估：{:.2f}'.format(clf.score(X_test,y_test)))
print('训练集评估：{:.2f}'.format(clf.score(X_train,y_train)))



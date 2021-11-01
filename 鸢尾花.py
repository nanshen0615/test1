# coding=utf-8
# @Time:2021/8/17 20:32
# @Author: 张泽阳
# @Email: 912984027@qq.com
from sklearn.datasets import load_iris  # sklearn库中自带iris
iris_datas=load_iris()  #鸢尾花数据集
#   1 数据的查看
# print(iris_datas.keys())   #key和value
# print(iris_datas['DESCR'][:193]+"\n...")  #数据集的简要说明
# print(iris_datas['target_names']) #品种\类型 ['setosa' 'versicolor' 'virginica']
# print(iris_datas['feature_names'])#特征['sepal length (cm)花萼长', 'sepal width (cm)花萼宽', 'petal length (cm)花瓣长', 'petal width (cm)花瓣宽']
# print(iris_datas['data'].shape)#(150, 4)  150行 4列
# print(iris_datas['data'][:5]) #前五行的数据
# print(iris_datas['target']) #3个品种简称  0-2对应 ['setosa  0' 'versicolor  1' 'virginica  2']

#   2  建立训练数据（training data）和测试数据（text data）  自动train_test_split  比例75%训练数据和25%测试数据
from sklearn.model_selection import train_test_split
X_train,X_text,y_train,y_text=train_test_split(iris_datas['data'],iris_datas['target'],random_state=0)#  自动train_test_split  比例75%训练数据和25%测试数据  random_state指定的  保证函数的输出不变
print(X_train.shape) #75%
print(y_train.shape)
print(X_text.shape)  #25%
print(y_text.shape)  #shape读取矩阵1\2维矩阵 shape[长,宽]
from sklearn.neighbors import KNeighborsClassifier  #近邻分类算法
knn=KNeighborsClassifier(n_neighbors=1)  #选取最进的3-5个邻居  而不是只考虑最近的一个  *
knn.fit(X_train,y_train)

#   3做预测    发现一朵新花
import numpy as np
X_new=np.array([5,2.9,1,0.2])  #新花的数据

#  4评估模型
y_pred=knn.predict(X_text) #计算精度 看他属于那个类型
print(np.mean(y_pred==y_text))  #查看精度0.9736842105263158
print(knn.score(X_text,y_text))  #查看精度0.9736842105263158
 #1：导入模块    2：花的种类（3类）  3：列头的名称  4：文件的行和列  5：品种的shape转换0-2  6：建立训练数据75%和验证数据25%  7：实例化对象（KNeighborsClassifier）、拟合  8：做预测  9：评估精度
# 需要先引进包 调用了sklearn包的svm类，用于后续程序的分类器训练
from sklearn import svm

X = [[3, 3], [1, 1]]  # 定义两个训练样本的特征向量
y = [3, 9]  # 是与 X 中特征向量对应的类标签
clf = svm.SVC()  # 定义用于分类的 svm 的分类器
clf. fit(X, y)  # 是对以 X 作为特征向量 y 作为类标 tag 的样本数据 进行有监督训练

ans = clf.predict([[2., 2.]])  # 用来预测[2., 2.] 属于哪一类 并将预测结果返回给ans
print(ans)





from sklearn import svm  # 需要先引进包 调用了sklearn包的svm类，用于后续程序的分类器训练
X = [[3, 3], [4, 4]]  # 定义两个训练样本的特征向量
y = [3, 2]  # 是与 X 中特征向量对应的类标签
clf = svm.SVC()  # 定义用于分类的 svm 的分类器
clf. fit(X, y)  # 是对以 X 作为特征向量 y 作为类标 tag 的样本数据 进行有监督训练

# 用来预测[8., 8.] 属于哪一类 并将预测结果返回给ans  biru 3.3  是奇数 ， 4 4 是偶数 4 4 的 tag 是  y = [3, 2] 的 2
# 所以最后打印出来是  [2]
ans = clf.predict([[8., 8.]])
print(ans)


# X = [[3, 3], [1, 1]]
# y = [3, 9]
# clf = svm.SVC()
# clf. fit(X, y)
# # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# #   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
# #   max_iter=-1, probability=False, random_state=None, shrinking=True,
# #   tol=0.001, verbose=False)
# ans = clf.predict([[2., 2.]])
# print(ans)
# # [9]
# y = [2, 7]
# clf = svm.SVC()
# clf. fit(X, y)
# # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# #   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
# #   max_iter=-1, probability=False, random_state=None, shrinking=True,
# #   tol=0.001, verbose=False)
# ans = clf.predict([[2., 2.]])
# print(ans)
# # [7]
# X = [[2, 2], [1, 1]]
# y = [2, 7]
# clf = svm.SVC()
# clf. fit(X, y)
# # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# #   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
# #   max_iter=-1, probability=False, random_state=None, shrinking=True,
# #   tol=0.001, verbose=False)
# ans = clf.predict([[2., 2.]])
# print(ans)
# # [2]
# X = [[3, 3], [12, 12]]
# y = [0, 1]
# clf = svm.SVC()
# clf. fit(X, y)
# # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# #   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
# #   max_iter=-1, probability=False, random_state=None, shrinking=True,
# #   tol=0.001, verbose=False)
# ans = clf.predict([[5., 5.]])
# print(ans)
# # [0]
# X = [[1, 3], [2, 16]]
# y = [3, 4]
# clf = svm.SVC()
# clf. fit(X, y)
# # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# #   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
# #   max_iter=-1, probability=False, random_state=None, shrinking=True,
# #   tol=0.001, verbose=False)
# ans = clf.predict([[3., 81.]])
# print(ans)
# # [4]


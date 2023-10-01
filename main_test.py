from sklearn import datasets #导入数据集模块
from sklearn.model_selection import train_test_split #数据集划分
from sklearn import svm  #导入SVM支持向量机
from sklearn.metrics import classification_report   #用于显示主要分类指标的文本报告
import sklearn.metrics as sm #生成混淆矩阵的库
import seaborn as sn  #混淆矩阵可视化的库
import matplotlib.pyplot as plt #画图
#--------------------------------1.加载数据集---------------------------------#
iris = datasets.load_iris()#加载鸢尾花数据集
print(iris)
X = iris.data #输入特征
Y = iris.target #标签（输出特征）
print(X)
print('----------------')
print(Y)
print('----------------')
print('鸢尾花输入特征的维度是{}'.format(X.shape))
print('鸢尾花标签的维度是{}'.format(Y.shape))

#--------------------------------2.划分数据集---------------------------------#
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=45) # 数据划分
#--------------------------------3.模型训练---------------------------------#
clas = svm.SVC()#选择分类器
clas.fit(X_train,Y_train)#训练
#--------------------------------4.模型预测---------------------------------#
Y_pre=clas.predict(X_test)#预测测试集

#--------------------------------5.性能评估---------------------------------#
m = sm.confusion_matrix(Y_test, Y_pre) #生成混淆矩阵
print('混淆矩阵为：', m, sep='\n')
ax = sn.heatmap(m,annot=True,fmt='.20g')
ax.set_title('confusion matrix')
ax.set_xlabel('predict')
ax.set_ylabel('true')
plt.show() #混淆矩阵可视化
print("测试集准确率：%s"%clas.score(X_test,Y_test))   #输出测试集准确度
print("分析报告:",classification_report(Y_test,Y_pre))#生成分类报告

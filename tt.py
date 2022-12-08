import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
df=pd.read_csv('vgsales.csv')
#----------------------------------------------------------------------------------------------------
#特征分布
# result=df[['Global_Sales']].groupby(df['Platform']).sum()
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.barh(result.index,result['Global_Sales'])
# plt.show()

# result=df[['Global_Sales']].groupby(df['Genre']).sum()
# print(result)
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.barh(result.index,result['Global_Sales'])
# plt.show()

# result=df[['Global_Sales']].groupby(df['Publisher']).sum()
# res=result.sort_values(by='Global_Sales')
# print(res.tail(15))
# re=res.tail(15)
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.barh(re.index,re['Global_Sales'])
# plt.show()

# re=df[['Global_Sales']].groupby(df['Year']).sum()
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.barh(re.index,re['Global_Sales'])
# plt.show()

# re=df[['Global_Sales']].groupby(df['NA_Sales']).sum()
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.bar(re.index,re['Global_Sales'])
# plt.show()

# re=df[['Global_Sales']].groupby(df['EU_Sales']).sum()
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.bar(re.index,re['Global_Sales'])
# plt.show()

# re=df[['Global_Sales']].groupby(df['JP_Sales']).sum()
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.bar(re.index,re['Global_Sales'])
# plt.show()

# re=df[['Global_Sales']].groupby(df['Other_Sales']).sum()
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.bar(re.index,re['Global_Sales'])
# plt.show()

# data=df.pivot_table(index='Genre',
#                     values=['JP_Sales','EU_Sales',
#                     'NA_Sales','Global_Sales'],
#                     aggfunc=np.sum)

# data['NA_prop']=data['NA_Sales']/data['Global_Sales']
# data['JP_prop']=data['JP_Sales']/data['Global_Sales']
# data['EU_prop']=data['EU_Sales']/data['Global_Sales']

# f,ax=plt.subplots(figsize=(12,8))
# index=np.arange(len(data))
# minColor = (31/256,78/256,95/256)   
# midColor = (121/256,168/256,169/256)  
# maxColor = (170/256,207/256,208/256) 
# plt.bar(index,data.NA_prop,color=minColor)
# plt.bar(
#         index,data.JP_prop,
#         bottom=data.NA_prop, 
#         color=midColor
#         )
# plt.bar(
#         index,data.EU_prop,
#         bottom=data.NA_prop, 
#         color=maxColor
#         )
# font={
#     'family':'DejaVu Sans',
#     'weight':'normal',
#     'size':12
# }
# plt.xticks(index,data.index,rotation=90,fontsize=6)
# plt.title('The Proportion of Different Areas',font)
# plt.ylabel('Proportion',font)
# plt.legend(['NA_Sales','JP_Sales','EU_Sales'],
# loc='upper center',ncol=3,framealpha=0.6)
# plt.show()
# dfiris=pd.read_csv('iris_training.csv')
# fig=plt.figure('Iris Data',figsize=(30,30))
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# column_names=['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
# iris=np.array(dfiris)
# for i in range(4):
#     for j in range(4):
#         plt.subplot(4,4,4*i+(j+1))
#         if(i==j):
#             plt.text(0.3,0.4,column_names[i],fontsize=6)
#         else:
#             plt.scatter(iris[:,j],iris[:,i],c=iris[:,4],cmap='winter_r')
            
#         if(i==0):
#             plt.title(column_names[j],fontsize=6)
#         if(j==0):
#             plt.ylabel(column_names[i],fontsize=6)
# plt.show()
#相关性系数矩阵
# corrdf=df.corr()
# print('相关性系数：')
# print(corrdf)
# print(corrdf['Global_Sales'].sort_values(ascending=False))
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False
# plt.subplots(figsize=(9, 9))
# sns.heatmap(corrdf, annot=True, vmax=1, square=True, cmap="Blues")
# plt.show()
# tf=pd.read_csv('iris_training.csv')
# corrdf=tf.corr()
# print('相关性系数：')
# print(corrdf)
# print(corrdf['Species'].sort_values(ascending=False))
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False
# plt.subplots(figsize=(9, 9))
# sns.heatmap(corrdf, annot=True, vmax=1, square=True, cmap="Blues")
# plt.show()
#----------------------------------------------------------------------------------------------------
#因子分析
# Bartlett's球状检验
# df.dropna(inplace=True)
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
# re=df.drop(['Platform','Publisher','Genre','Year'],axis=1)
# chi_square_value, p_value = calculate_bartlett_sphericity(re)
# print(chi_square_value)
# print(p_value)

#KMO检验
# from factor_analyzer.factor_analyzer import calculate_kmo
# df.dropna(inplace=True)
# re=df.drop(['Platform','Publisher','Genre','Year'],axis=1)
# kmo_all,kmo_model=calculate_kmo(re)
# print(kmo_model)

# 选择因子个数
# df.dropna(inplace=True)
# faa = FactorAnalyzer(25,rotation=None)
# re=df.drop(['Platform','Publisher','Genre','Year'],axis=1)
# faa.fit(re)
# ev,v=faa.get_eigenvalues()
#
# plt.scatter(range(1, re.shape[1] + 1), ev)
# plt.plot(range(1, re.shape[1] + 1), ev)
# plt.title("Scree Plot")  
# plt.xlabel("Factors")
# plt.ylabel("Eigenvalue")
# plt.grid()  
# plt.show() 

#因子旋转
# df.dropna(inplace=True)
# re=df.drop(['Platform','Publisher','Genre','Year'],axis=1)
# faa_two = FactorAnalyzer(2,rotation='varimax')
# faa_two.fit(re)
# print(faa_two.fit(re))
# print(faa_two.get_communalities())
# print(pd.DataFrame(faa_two.get_communalities(),index=re.columns))
# print(pd.DataFrame(faa_two.get_eigenvalues()))
# print(pd.DataFrame(faa_two.loadings_,index=re.columns))
# print(faa_two.get_factor_variance())
# df1 = pd.DataFrame(np.abs(faa_two.loadings_),index=re.columns)
# print(df1)
# plt.figure(figsize = (14,14))
# ax = sns.heatmap(df1, annot=True, cmap="Blues")
# ax.yaxis.set_tick_params(labelsize=6)
# plt.title("Factor Analysis", fontsize="xx-large")
# plt.ylabel("Sepal Width", fontsize="xx-large")
# plt.show()
# df2 = pd.DataFrame(faa_two.transform(re))
# print(df2)
#----------------------------------------------------------------------------------------------------
#回归分析
#tf=pd.read_csv('iris_training.csv')
# sns.pairplot(tf, x_vars=['Species','SepalLength','SepalWidth','PetalLength','PetalWidth'], y_vars='Species', size=7, aspect=0.8,kind = 'reg')
# plt.show()
#X_train,X_test,Y_train,Y_test = train_test_split(tf.iloc[:,:4],tf.Species,train_size=.80)
# print("原始数据特征:",tf.iloc[:,:4].shape,
#       ",训练数据特征:",X_train.shape,
#       ",测试数据特征:",X_test.shape)
 
# print("原始数据标签:",tf.Species.shape,
#       ",训练数据标签:",Y_train.shape,
#       ",测试数据标签:",Y_test.shape)
# model = LinearRegression()
# model.fit(X_train,Y_train)
# a  = model.intercept_
# b = model.coef_
#print(a,b)
#对线性回归进行预测
# score = model.score(X_test,Y_test)
# print(score)
# Y_pred = model.predict(X_test)
# print(Y_pred)
# plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
# plt.show()
#---------------------------------------------------------------------------------------------------------
#主成分分析
# Bartlett's球状检验
# tf=pd.read_csv('iris_training.csv')
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
# chi_square_value, p_value = calculate_bartlett_sphericity(tf)
# print(chi_square_value, p_value)
# KMO检验
# from factor_analyzer.factor_analyzer import calculate_kmo
# kmo_all, kmo_model = calculate_kmo(tf)
# print(kmo_all)

# from sklearn import preprocessing
# zf = preprocessing.scale(tf)
#print(zf)
# covX = np.around(np.corrcoef(zf.T),decimals=3)
# featValue, featVec= np.linalg.eig(covX.T)  
#print(featValue, featVec)
# 同样的数据绘制散点图和折线图
# plt.scatter(range(1, zf.shape[1] + 1), featValue)
# plt.plot(range(1, zf.shape[1] + 1), featValue)
# plt.title("Scree Plot")  
# plt.xlabel("Factors")
# plt.ylabel("Eigenvalue")
# plt.grid()  
# plt.show()  
# gx = featValue/np.sum(featValue)
# lg = np.cumsum(gx)
# print(lg)
# #选出主成分
# k=[i for i in range(len(lg)) if lg[i]<0.96]
# k = list(k)
# print(k)
# selectVec = np.matrix(featVec.T[k]).T
# selectVe=selectVec*(-1)
# print(selectVec)
# finalData = np.dot(zf,selectVec)
# print(finalData)
# # 绘图
# plt.figure(figsize = (14,14))
# ax = sns.heatmap(selectVec, annot=True, cmap="Blues")
# ax.yaxis.set_tick_params(labelsize=6)
# plt.title("Factor Analysis", fontsize="xx-large")
# plt.ylabel("Sepal Width", fontsize="xx-large")
# plt.show()
#主成分回归
# from sklearn.decomposition import PCA
# x=zf[:,:-1]
# y=zf[:,-1]
#训练pca模型
# model_pca=PCA()
# data_pca=model_pca.fit_transform(x)
# ratio_cs=np.cumsum(model_pca.explained_variance_ratio_)  
# rule_index=np.where(ratio_cs>0.85)
# index=rule_index[0][0]  
# data_pca_result=data_pca[:,:index+1]   
# model_linear=LinearRegression()  
# model_linear.fit(data_pca_result,y)  
# print(model_linear.coef_ )
# print(model_linear.intercept_ ) 

#------------------------------------------------------------------------------------------------------------
#判别分析
#Fisher
#-------------------------------------------------------
# def Fisher(X1,X2,n,c):
    
#     m1=(np.mean(X1,axis = 0))
#     m2=(np.mean(X2,axis = 0))
#     m1 = m1.reshape(n,1)   
#     m2 = m2.reshape(n,1)

#     #计算类内离散度矩阵
#     S1 = np.zeros((n,n))              
#     S2 = np.zeros((n,n))              
#     if c == 0:                       
#         for i in range(0,49):
#             S1 += (X1[i].reshape(n,1)-m1).dot((X1[i].reshape(n,1)-m1).T)
#         for i in range(0,50):
#             S2 += (X2[i].reshape(n,1)-m2).dot((X2[i].reshape(n,1)-m2).T)
#     if c == 1:
#         for i in range(0,50):
#             S1 += (X1[i].reshape(n,1)-m1).dot((X1[i].reshape(n,1)-m1).T)
#         for i in range(0,49):
#             S2 += (X2[i].reshape(n,1)-m2).dot((X2[i].reshape(n,1)-m2).T)
#     #计算总类内离散度矩阵S_w
#     S_w = S1 + S2

#     #计算最优投影方向 W
#     W = np.linalg.inv(S_w).dot(m1 - m2)
#     #在投影后的一维空间求两类的均值
#     m_1 = (W.T).dot(m1)
#     m_2 = (W.T).dot(m2)
    
#     #计算分类阈值 W0(为一个列向量)
#     W0 = -0.5*(m_1 + m_2)
    
#     return W,W0

# def Classify(X,W,W0):
#     y = (W.T).dot(X) + W0
#     return y

# iris = pd.read_csv('iris.data',header=None,sep=',')
# iris1 = iris.iloc[0:150,0:4]
# iris2 = np.mat(iris1)

# Accuracy = 0
# accuracy_ = np.zeros(10)
   

# P1 = iris2[0:50,0:4]
# P2 = iris2[50:100,0:4]
# P3 = iris2[100:150,0:4]
    
# G121 = np.ones(50)
# G122 = np.ones(50)
# G131 = np.zeros(50)
# G132 = np.zeros(50)
# G231 = np.zeros(50)
# G232 = np.zeros(50)
   
# # 留一法验证准确性
# # 第一类和第二类的线性判别
# count = 0
# for i in range(100):
#     if i <= 49:
#         test = P1[i]
#         test = test.reshape(4,1)
#         train = np.delete(P1,i,axis=0)     
#         W,W0 = Fisher(train,P2,4,0)
#         if (Classify(test,W,W0)) >= 0:
#             count += 1
#             G121[i] = Classify(test,W,W0)
#     else:
#         test = P2[i-50]
#         test = test.reshape(4,1)
#         train = np.delete(P2,i-50,axis=0)
#         W,W0 = Fisher(P1,train,4,1)
#         if (Classify(test,W,W0)) < 0:
#             count += 1
#             G122[i-50] = Classify(test,W,W0)
# Accuracy12 = count/100
# print("第一类和二类的分类准确率为:%.3f"%(Accuracy12))

# # 第一类和第三类的线性判别
# count = 0
# for i in range(100):
#     if i <= 49:
#         test = P1[i]
#         test = test.reshape(4,1)
#         train = np.delete(P1,i,axis=0)      
#         W,W0 = Fisher(train,P3,4,0)
#         if (Classify(test,W,W0)) >= 0:
#             count += 1
#             G131[i] = Classify(test,W,W0)
#     else:
#         test = P3[i-50]
#         test = test.reshape(4,1)
#         train = np.delete(P3,i-50,axis=0)
#         W,W0 = Fisher(P1,train,4,1)
#         if (Classify(test,W,W0)) < 0:
#             count += 1
#             G132[i-50] = Classify(test,W,W0)

# Accuracy13 = count/100
# print("第一类和三类的分类准确率为:%.3f"%(Accuracy13))

# # 第二类和第三类的线性判别
# count = 0
# for i in range(100):
#     if i <= 49:
#         test = P2[i]
#         test = test.reshape(4,1)
#         train = np.delete(P2,i,axis=0)     
#         W,W0 = Fisher(train,P3,4,0)
#         if (Classify(test,W,W0)) >= 0:
#             count += 1
#             G231[i] = Classify(test,W,W0)
#     else:
#         test = P3[i-50]
#         test = test.reshape(4,1)
#         train = np.delete(P3,i-50,axis=0)
#         W,W0 = Fisher(P2,train,4,1)
#         if (Classify(test,W,W0)) < 0:
#             count += 1
#             G232[i-50] = Classify(test,W,W0)

# Accuracy23 = count/100
# print("第二类和三类的分类准确率为:%.3f"%(Accuracy23))

# # 画相关的图
# import matplotlib.pyplot as plt

# y1 = np.zeros(50)
# y2 = np.zeros(50)
# plt.figure(1)
# plt.ylim((-0.5,0.5))           
# #画散点图
# plt.scatter(G121, y1,c='red', alpha=1, marker='.')
# plt.scatter(G122, y2,c='k', alpha=1, marker='.')
# plt.xlabel('Class:1-2')



# plt.figure(2)
# plt.ylim((-0.5,0.5))            
# #画散点图
# plt.scatter(G131, y1,c='red', alpha=1, marker='.')
# plt.scatter(G132, y2,c='k', alpha=1, marker='.')
# plt.xlabel('Class:1-3')



# plt.figure(3)
# plt.ylim((-0.5,0.5))           
# #画散点图
# plt.scatter(G231, y1,c='red', alpha=1, marker='.')
# plt.scatter(G232, y2,c='k', alpha=1, marker='.')
# plt.xlabel('Class:2-3')
#plt.show()

#----------------------------------------------------------------------
#二次判别
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = pd.DataFrame(iris.data, columns=iris.feature_names)
# y = iris.target
# X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)
# model = QuadraticDiscriminantAnalysis()
# model.fit(X_train, y_train)
# model.score(X_test, y_test)    
# prob = model.predict_proba(X_test)
# prob[:3]
# pred = model.predict(X_test)
# pred[:5]
# confusion_matrix(y_test, pred)
# print(classification_report(y_test, pred))
# cohen_kappa_score(y_test, pred)
# from mlxtend.plotting import plot_decision_regions
# X2 = X.iloc[:, 2:4]
# model = QuadraticDiscriminantAnalysis()
# model.fit(X2, y)
# model.score(X2, y)
# plot_decision_regions(np.array(X2), y, model)
# plt.xlabel('petal_length')
# plt.ylabel('petal_width')
# plt.title('Decision Boundary for QDA')
# plt.show()

#---------------------------------------------------------------------------------------------------------------
#类平均法
# from scipy.cluster.hierarchy import dendrogram,linkage
# data=pd.read_csv('iris_training.csv')
# z=linkage(data,"average")
# #画图
# fig,ax=plt.subplots(figsize=(8,8))
# dendrogram(z,leaf_font_size=4)
# plt.show()
#最短距离法
# from scipy.cluster.hierarchy import dendrogram,linkage
# data=pd.read_csv('iris_training.csv')
# z=linkage(data,"single")
# #画图
# fig,ax=plt.subplots(figsize=(8,8))
# dendrogram(z,leaf_font_size=4)
# plt.show()
#最长距离法
# from scipy.cluster.hierarchy import dendrogram,linkage
# data=pd.read_csv('iris_training.csv')
# z=linkage(data,"complete")
# #画图
# fig,ax=plt.subplots(figsize=(8,8))
# dendrogram(z,leaf_font_size=4)
# plt.show()
#重心法
# from scipy.cluster.hierarchy import dendrogram,linkage
# data=pd.read_csv('iris_training.csv')
# z=linkage(data,"median")
# #画图
# fig,ax=plt.subplots(figsize=(8,8))
# dendrogram(z,leaf_font_size=4)
# plt.show()
#
# from scipy.cluster.hierarchy import dendrogram,linkage
# data=pd.read_csv('iris_training.csv')
# z=linkage(data,"ward")
# #画图
# fig,ax=plt.subplots(figsize=(8,8))
# dendrogram(z,leaf_font_size=4)
# plt.show()
#---------------------------------------------------------------------------------------------
#kmeans
# from sklearn import datasets
# from sklearn.cluster import KMeans
# def draw_result(train_x,labels,cents,title):
# 	n=np.unique(labels).shape[0]  
# 	color=['red','orange','yellow']
# 	plt.figure()
# 	plt.title(title)
# 	for i in range(n):
# 		current_data= train_x[labels==i]
# 		plt.scatter(current_data[:,0],current_data[:,1],c=color[i])
# 		plt.scatter(cents[i,0],cents[i,1],c='blue',marker='*',s=100)
# 	return plt
# iris = datasets.load_iris()
# iris_x=iris.data
# clf=KMeans(n_clusters=3,max_iter=10,n_init=10,init='k-means++',algorithm='full',tol=1e-4,random_state=1)
# clf.fit(iris_x)
# print('SSE=',clf.inertia_)
# draw_result(iris_x,clf.labels_,clf.cluster_centers_,'kmeans').show()






# Facebook 签到位置
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 一、处理数据以及特征工程
# 1、读取收，缩小数据的范围
data = pd.read_csv("/Users/liuzikai/Desktop/Pycharm/Machine- Learning/FBlocation/train.csv")
# test = "/Users/liuzikai/Desktop/Pycharm/Machine- Learning/FBlocation/test.csv"

# 数据逻辑筛选操作 df.query() -- 缩小数据范围节省时间 -- 以x,y 坐标作为基准
data = data.query("x > 2 & x < 2.5 & y > 1 & y < 1.5")

# 处理时间特征 -- time 一栏转换成有用数据
timevalue = pd.to_datetime(data['time'] , unit="s")
date = pd.DatetimeIndex(timevalue)
data["day"]=date.day
data["weekday"]=date.weekday
data["hour"]=date.hour

# 3 过滤签到次数少的地点 -- 统计place_id 出现次数 -- 出现次数大于3
place_count = data.groupby("place_id").count()["row_id"]
# print(place_count[place_count > 3].head())
data_final = data[data["place_id"].isin(place_count[place_count > 3].index.values)]

# 筛选特征值和目标值
x = data_final[["x","y","accuracy","day","weekday","hour"]]
y = data_final["place_id"]

# 数据集划分
x_train, x_test,y_train, y_test = train_test_split(x,y)
# print(x_train)

# 3）特征工程：标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 4）KNN算法预估器
estimator = KNeighborsClassifier()

# 加入网格搜索与交叉验证
# 参数准备
param_dict = {"n_neighbors": [3, 5, 7, 9]}
estimator = GridSearchCV(estimator,param_grid=param_dict,cv=3)
estimator.fit(x_train,y_train)

# 5）模型评估
# 方法1：直接比对真 实值和预测值
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("直接比对真实值和预测值:\n", y_test == y_predict)
# 方法2：计算准确率 -- 整体测试集效果
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)

# 最佳参数：best_params_
print("最佳参数：\n", estimator.best_params_)
# 最佳结果：best_score_ -- 验证集结果
print("最佳结果：\n", estimator.best_score_)
# 最佳估计器：best_estimator_
print("最佳估计器:\n", estimator.best_estimator_)
# 交叉验证结果：cv_results_
print("交叉验证结果:\n", estimator.cv_results_)




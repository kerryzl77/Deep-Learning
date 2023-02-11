import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# 案例：探究用户对物品类别的喜好细分
# 用户 user_id 物品类别 aisle
# 1）需要将user_id和aisle放在同一个表中 - 合并
# 2）找到user_id和aisle - 交叉表和透视表
# 3）特征冗余过多 -> PCA降维

# prior表中有：product_id ,order_id
# products表中： product_id ,aisle_id
# orders表中有：order_id ,user_id
# aisles表中： aisles_id,aisle

# 1）获取数据
aisles = pd.read_csv("/instacart-market-basket-analysis/aisles.csv")
products = pd.read_csv("/instacart-market-basket-analysis/products.csv")
order_products = pd.read_csv("/instacart-market-basket-analysis/order_products__prior.csv")
orders = pd.read_csv("/instacart-market-basket-analysis/orders.csv")

# 1）合并表
tab1 = pd.merge(aisles,products,on=["aisle_id","aisle_id"])
tab2 = pd.merge(tab1,order_products,on=["product_id","product_id"])
tab3 = pd.merge(tab2,orders,on=["order_id","order_id"])
# print(tab3.keys())

# 2）找到user_id和aisle - 交叉表和透视表
table = pd.crosstab(tab3['user_id'],tab3['aisle'])
data = table[:10000]
# print(table.head())

# 3）特征冗余过多 -> PCA降维
transfer = PCA(n_components=0.95)
data_new = transfer.fit_transform(data)
print(data_new.shape)

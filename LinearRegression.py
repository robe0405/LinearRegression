import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston # データセット
from sklearn.model_selection import train_test_split # データの分割
from sklearn.linear_model import LinearRegression # 線形回帰

dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["PRICE"] = dataset.target #正解ラベルはtargetに入っている
print(df)

"""
#欠損値の確認
print(df.isnull().any())
"""

# データ全体の散布図(相関関係を簡単に見極める)
pd.plotting.scatter_matrix(df)
plt.show()

X = df.loc[:, ["RM"]].values # array配列
Y = df.loc[:, ["PRICE"]].values # array配列

# 学習
model = LinearRegression()
model.fit(X,Y)

print('coefficient = ', model.coef_[0]) # 説明変数の係数を出力
print('intercept = ', model.intercept_) # 切片を出力

plt.scatter(X, Y) # 散布図
plt.xlabel("RM")
plt.ylabel("PRICE")
plt.title("boston")
plt.legend()
plt.plot(X, model.predict(X), color='red') # 回帰直線
plt.show()

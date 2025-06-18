import numpy as np
import pandas as pd

data = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]

# 创建DataFrame
df = pd.DataFrame(data, columns=['Site', 'Age'])

# 使用astype方法设置每列的数据类型
df['Site'] = df['Site'].astype(str)
df['Age'] = df['Age'].astype(float)

print(df)
print("\n")
print(df.loc[0])
print("\n")
print(df.iloc[1])
print("\n")

print (pd.DataFrame({'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}))

print(pd.DataFrame(np.array([
    ['Google', 10],
    ['Runoob', 12],
    ['Wiki', 13]
]), columns=['Site', 'Age']))


print(pd.DataFrame([{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]))
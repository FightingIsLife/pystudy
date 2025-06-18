import pandas as pd

# 示例数据
data = {
    'Height': [150, 160, 170, 180, 90],
    'Weight': [45, 55, 65, 75, 15],
    'Age': [20, 25, 30, 35, 40]
}

df = pd.DataFrame(data)

# 计算皮尔逊相关系数
correlation = df.corr(method='pearson')
print(correlation)
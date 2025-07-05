import pandas as pd

# 示例数据
data = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [45, 55, 65, 75, 75],
    'Age': [20, 25, 30, 35, 31]
}

df = pd.DataFrame(data)

# 计算斯皮尔曼等级相关系数
spearman_correlation = df.corr(method='spearman')
print(spearman_correlation)
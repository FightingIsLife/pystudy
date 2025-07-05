import pandas as pd

# 示例数据
data = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [45, 55, 65, 75, 85],
    'Age': [20, 25, 30, 35, 40]
}

df = pd.DataFrame(data)

# 计算肯德尔秩相关系数
kendall_correlation = df.corr(method='kendall')
print(kendall_correlation)
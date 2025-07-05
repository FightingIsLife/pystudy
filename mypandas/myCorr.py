import pandas as pd

# 示例数据
data = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [45, 55, 65, 75, 75],
    'Age': [20, 25, 30, 35, 31]
}
df = pd.DataFrame(data)

# 计算相关性矩阵
correlation_matrix = df.corr()
print(correlation_matrix)
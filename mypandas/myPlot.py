import pandas as pd
import matplotlib.pyplot as plt

# 示例数据
data = {'Year': [2015, 2016, 2017, 2018, 2019, 2020],
        'Sales': [100, 150, 200, 250, 300, 250]}
df = pd.DataFrame(data)

# 绘制折线图
df.plot(kind='line', x='Year', y='Sales', title='Sales Over Years', xlabel='Year', ylabel='Sales', figsize=(10, 6))
plt.show()
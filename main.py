import pandas as pd

# 模拟订单数据（类似Java的POJO集合）
orders = pd.DataFrame({
    'order_id': [1001, 1002, 1003, 1004, 1005, 1006],
    'user_id': [101, 102, 101, 103, 102, 104],
    'product': ['A', 'B', 'A', 'C', 'B', 'A'],
    'amount': [300, 450, 200, 150, 500, 350],
    'city': ['北京', '上海', '北京', '广州', '上海', '北京'],
    'order_date': ['2023-01-10', '2023-01-15', '2023-01-20',
                   '2023-01-25', '2023-01-28', '2023-02-01']
})

print("原始数据集：")
print(orders)


# 添加日期列（Java风格：字符串转日期）
orders['order_date'] = pd.to_datetime(orders['order_date'])

# 修复方案1：保持原始顺序
orders['orig_index'] = orders.index
orders['rolling_avg'] = (
    orders.sort_values('orig_index')
    .groupby('city')['amount']
    .rolling(window=2, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .sort_index()
)

# 修复方案2：transform更简洁
orders['rolling_avg_transform'] = (
    orders.groupby('city')['amount']
    .transform(lambda x: x.rolling(window=2, min_periods=1).mean())
)


# ====== 你的代码 ======

print("\n挑战3结果：")
print(orders)
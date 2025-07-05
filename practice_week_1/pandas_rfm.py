import pandas as pd
data = {
    'customer_id': [101, 102, 101, 103, 102, 104],
    'order_date': pd.to_datetime(['2023-01-05', '2023-01-12', '2023-01-18', '2023-01-03', '2023-01-22', '2023-01-10']),
    'amount': [120, 80, 150, 90, 110, 60]
}
orders = pd.DataFrame(data)
current_date = pd.Timestamp('2023-01-25')

# 请实现：
# 1. 计算每个客户的 Recency（最近一次购买日期到当前日期的天数）、Frequency（购买次数）和 Monetary（总购买金额）

orders['days'] = (current_date - orders['order_date']).dt.days  # 转换为整数天数

r = orders.groupby('customer_id').agg(
    Recency=('days', 'min'),
    Frequency=('customer_id', 'count'),
    Monetary=('amount', 'sum')
).reset_index()  # 重置索引将customer_id转换为列

# 打印结果
print(r)

# 2. 输出Recency < 15天 且 Monetary > 100的客户ID列表

condition = (r['Recency'] < 15) & (r['Monetary'] > 100)
print(r[condition])

customer_ids = r[condition].index.tolist()

print(customer_ids)
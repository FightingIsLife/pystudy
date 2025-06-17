import numpy as np

# 原始价格表（3种商品×5个城市）
prices = np.array([
    [100, 120, 130],  # 北京
    [110, 125, 135],  # 上海
    [105, 118, 128],  # 广州
    [115, 122, 140],  # 深圳
    [108, 116, 132]   # 杭州
])

# 城市专属折扣率（复用Java枚举经验）
discounts = np.array([0.9, 0.85, 0.88, 0.92, 0.95])  # 各城市折扣率

# 广播计算（折扣后价格）
discounted = prices * discounts[:, np.newaxis]  # 关键广播操作

print("原始价格：\n", prices)
print("折扣后价格：\n", discounted.round(4))


# 建立布尔索引（类似Java的Predicate）
high_profit_mask = discounted > 110

# 花式索引（商品1和商品3的上海/广州数据）
selected = discounted[[1,2], [0,2]]  # 索引组合：上海商品1，广州商品3

# 索引技巧实战 - 找出价格异常点
q25, q75 = np.percentile(discounted, [25, 75], axis=0)
iqr = q75 - q25
outliers = (discounted < q25 - 1.5*iqr) | (discounted > q75 + 1.5*iqr)



# # 金融数据正态化
# returns = np.random.normal(0.02, 0.05, 100)  # 模拟100日收益率
#
# # 关键计算函数
# print("收益均值和标准差：", np.mean(returns), np.std(returns))
# print("最大回撤：", np.min(returns - np.maximum.accumulate(returns)))
# print("相关性计算：", np.corrcoef(prices[:,0], prices[:,1])[0,1])


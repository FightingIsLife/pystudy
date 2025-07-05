import numpy as np
import pandas as pd
import scipy as sp
import re
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# 一、基础数据处理

# 1.加载JSON数据到DataFrame，并显示数据结构信息

moments = pd.read_json("../big_json/moment.json", orient="records")

# 2.将createTime从毫秒时间戳转换为日期时间格式
moments.createTime = pd.to_datetime(moments.createTime, unit="ms")

# 3.创建新列content_length表示文本内容长度（字符数）
moments['content_length'] = moments['text'].str.len()

# 4.提取pics数组中图片数量到新列pic_count
moments['pic_count'] = moments['pics'].str.len()

# 5.过滤出所有status为2的动态记录
filtered = moments.loc[moments.status == 2]

# 二、统计分析
# 6.计算各地区(area)动态数量的分布比例
areaG = moments.groupby('area').agg(Count=('id', 'count')).reset_index()
areaG['rate'] = areaG['Count'] / len(moments)

# 7.找出likeCount和commentCount相关系数
rs = sp.stats.pearsonr(moments.likeCount, moments.commentCount)[0]

# 8.按type分组计算平均点赞数和平均评论数
typeAvg = moments.groupby('type').agg(AvgLikeCount=('likeCount', 'mean'),
                                      AvgCommentCount=('commentCount', 'mean')).reset_index()

# 9.统计不同图片数量(pic_count)的动态占比
picG = moments.groupby('pic_count').agg(Count=('id', 'count')).reset_index()
picG['rate'] = picG['Count'] / len(moments)

# 10.找出点赞量超过该类型平均点赞数的动态记录
likeMean = moments.groupby('type')['likeCount'].transform('mean')
likeFiltered = moments[moments['likeCount'] > likeMean]

#  三、时间序列分析
# 11.按日期聚合每天发布的动态数量
moments['date'] = moments['createTime'].dt.date
dateG = moments.groupby('date')['id'].count()

# 12.计算30天内各小时的动态发布密度分布
now = datetime.now()
recent_moments = moments[moments['createTime'] > now - pd.Timedelta(days=30)].copy()
recent_moments['hour'] = recent_moments['createTime'].dt.hour
hourly_distribution = recent_moments['hour'].value_counts(normalize=True).sort_index()

# 13.找出所有在周末（周六/日）发布的动态
weekendMs = moments[moments['createTime'].dt.dayofweek.isin([5, 6])]

# 14.计算每个用户(uid)首次和末次发动态的时间间隔
uid_interval = moments.groupby('uid')['createTime'].agg(['min', 'max'])
uid_interval['interval'] = (uid_interval['max'] - uid_interval['min']).dt.total_seconds()

# 15.创建时间序列：每天互动的总次数（点赞+评论）
moments['ex_active'] = moments.likeCount + moments.commentCount
dateExActiveG = moments.groupby('date').agg(ExActive=('ex_active', 'sum')).reset_index()


# 四、高级处理
# 16.对文本内容进行向量化：将text列转换为小写并删除所有emoji
# 定义删除 emoji 的函数
def remove_emoji(text):
    # Emoji 范围的正则表达式
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


moments['text_vector'] = moments['text'].apply(lambda x: remove_emoji(x.lower()))

# 17.识别文本长度异常值：<5字符或>100字符的记录
text_length_condition = (moments['content_length'] < 5) | (moments['content_length'] > 100)
error_text_length_moments = moments[text_length_condition]

# 18.创建复合指标：互动指数 = (likeCount × 0.7) + (commentCount × 0.3)
moments['ex2_active'] = moments.likeCount * 0.7 + moments.commentCount * 0.3

# 19.检测高价值用户：发布动态≥3条且平均互动指数≥10的用户
ex2_activeG = moments.groupby('uid')['ex2_active'].agg(['count', 'mean']).reset_index()
high_ex2_active_user = ex2_activeG[(ex2_activeG['mean'] >= 10) & (ex2_activeG['count'] >= 3)]

# 20.实现地区-类型交叉分析表，含边际统计值
cross_tab = pd.crosstab(moments.area, moments.type, margins=True, margins_name='总计')

# 五、Numpy专项
# 21.将likeCount和commentCount转换为Numpy数组
numpyArray = moments[['likeCount', 'commentCount']].to_numpy()

# 22.计算互动量（点赞+评论）的百分位数(25%, 50%, 75%)
interaction_sum = numpyArray[:, 0] + numpyArray[:, 1]
percentiles = np.percentile(interaction_sum, [25, 50, 75])

# 23.对互动量进行Z-score标准化处理
interaction_zscore = sp.stats.zscore(interaction_sum)
# ✅ 使用scipy.stats.zscore正确，但也可用纯Numpy实现：
# zscores = (interaction_sum - interaction_sum.mean()) / interaction_sum.std()


# 24.用向量化操作生成互动强度标签： 低互动：[0,5)  中互动：[5,20)  高互动：[20+]
# labels = np.zeros_like(interaction_sum)
# labels[(interaction_sum >= 5) & (interaction_sum < 20)] = 1
# labels[interaction_sum >= 20] = 2
conditions = [
    interaction_sum < 5,
    (interaction_sum >= 5) & (interaction_sum < 20),
    interaction_sum >= 20
]
choices = [0, 1, 2]  # 对应低、中、高
labels = np.select(conditions, choices, default=-1)  # 默认值防止遗漏

# 25.实现矩阵运算：用户-类型互动矩阵（行：用户uid，列：动态类型type，值：总互动量）
user_type_matrix = moments.pivot_table(index='uid', columns='type', values='ex_active', aggfunc='sum', fill_value=0)

# 六、业务分析场景
# 27.识别高潜力动态：低点赞但有高评论量的记录
avg_like = moments.likeCount.mean()
avg_comment = moments.commentCount.mean()
low_like_threshold = np.percentile(moments.likeCount, 25)
high_comment_threshold = np.percentile(moments.commentCount, 75)
high_potential = moments[
    (moments.likeCount <= low_like_threshold) &
    (moments.commentCount >= high_comment_threshold)
]
high_potential = high_potential.assign(
    engagement_ratio = lambda x: x.commentCount / (x.likeCount + 0.1)  # 避免除零
).sort_values('engagement_ratio', ascending=False)


# 28.找出相同用户短时间内（<1小时）的多条动态
# 更好的方法是同时检查前后记录
moments2 = moments.copy().sort_values(['uid', 'createTime'])
moments2['prev_time'] = moments2.groupby('uid')['createTime'].shift()
moments2['next_time'] = moments2.groupby('uid')['createTime'].shift(-1)

# 检查与前一条或后一条的时间差
close_mask = (
    ((moments2['createTime'] - moments2['prev_time']).dt.total_seconds() < 3600) |
    ((moments2['next_time'] - moments2['createTime']).dt.total_seconds() < 3600)
)
close_moments = moments2[close_mask]


# 29.分析内容长度与互动量的关系：分组：0-10字，11-30字，31-50字，50+字
len_conditions = [
    moments['content_length'] <= 10,
    (moments['content_length'] > 10) & (moments['content_length'] <= 30),
    (moments['content_length'] > 30) & (moments['content_length'] <= 50),
    moments['content_length'] > 50
]
len_choices = [1,2,3,4]
moments['len_labels'] = np.select(len_conditions, len_choices)
# 计算各分组的平均互动量（点赞+评论）
len_analysis = moments.groupby('len_labels').agg(
    avg_like=('likeCount', 'mean'),
    avg_comment=('commentCount', 'mean'),
    avg_interaction=('ex_active', 'mean'),
    count=('id', 'count')
).reset_index()

# 30.检测异常模式：包含多图(≥3)但零互动的动态
exception_ms = moments[(moments['pic_count']  >= 3) & (moments['ex_active'] == 0)&
    (moments.content_length < 10)]



# 31.设计用户参与度评分模型（需说明特征选择）
"""
用户参与度评分模型

特征选择：
1. 内容吸引力指标
   - 平均点赞数（核心指标）
   - 平均评论数（用户深度参与）
   - 图文动态占比（图越多参与度越高）

2. 用户活跃度指标
   - 发布频率（每周动态数）
   - 最近活跃时间（距今天数）
   - 密集动态比例（28题结果）

3. 内容质量指标
   - 平均文本长度
   - 高互动动态占比
   - 异常动态比例（17&30题结果）

模型构建：
评分 = (0.4 × 平均点赞标准化) + 
       (0.3 × 平均评论标准化) + 
       (0.2 × 图文动态占比) + 
       (0.1 × 发布频率标准化)

实施步骤：
1. 计算每个用户的基础指标：
   user_stats = moments.groupby('uid').agg(
        avg_like=('likeCount', 'mean'),
        avg_comment=('commentCount', 'mean'),
        pic_ratio=('pic_count', lambda x: (x > 0).mean()),
        post_freq=('createTime', lambda x: len(x)/ ((x.max() - x.min()).days if len(x) > 1 else 1))
    )

2. 对所有指标进行min-max标准化
3. 按公式计算综合评分
4. 按评分排序用户
"""


# 1. 计算基础用户指标
def calculate_user_engagement(df):
    # 计算每个用户的基础数据
    user_stats = df.groupby('uid').agg(
        total_moments=('id', 'count'),
        avg_like=('likeCount', 'mean'),
        avg_comment=('commentCount', 'mean'),
        avg_pics=('pic_count', 'mean'),
        first_date=('createTime', 'min'),
        last_date=('createTime', 'max'),
        median_length=('content_length', 'median'),
        high_eng_ratio=('ex_active', lambda x: (x > 10).mean())
    ).reset_index()

    # 计算持续时间（周数）
    user_stats['account_weeks'] = (user_stats['last_date'] - user_stats['first_date']).dt.days / 7 + 1

    # 计算发布频率（每周动态数）
    user_stats['post_freq'] = user_stats['total_moments'] / user_stats['account_weeks']

    # 计算含图动态占比
    user_stats['pic_ratio'] = user_stats['avg_pics'].apply(lambda x: 1 if x > 0 else 0)

    return user_stats.drop(columns=['first_date', 'last_date'])


# 2. 数据预处理
def preprocess_engagement_data(user_stats):
    # 选择特征列
    features = user_stats[['avg_like', 'avg_comment', 'post_freq', 'high_eng_ratio']]

    # 处理缺失值
    features = features.fillna(features.median())

    # Min-Max标准化
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # 创建标准化后的DataFrame
    scaled_df = pd.DataFrame(
        scaled_features,
        columns=['avg_like_scaled', 'avg_comment_scaled', 'post_freq_scaled', 'high_eng_scaled']
    )

    # 合并回用户信息
    return pd.concat([user_stats[['uid', 'pic_ratio']], scaled_df], axis=1)


# 3. 计算参与度评分
def calculate_engagement_score(user_data):
    """基于业务规则计算参与度评分"""
    weights = {
        'avg_like_scaled': 0.35,
        'avg_comment_scaled': 0.30,
        'post_freq_scaled': 0.15,
        'high_eng_scaled': 0.10,
        'pic_ratio': 0.10
    }

    # 计算加权评分
    user_data['engagement_score'] = (
            user_data['avg_like_scaled'] * weights['avg_like_scaled'] +
            user_data['avg_comment_scaled'] * weights['avg_comment_scaled'] +
            user_data['post_freq_scaled'] * weights['post_freq_scaled'] +
            user_data['high_eng_scaled'] * weights['high_eng_scaled'] +
            user_data['pic_ratio'] * weights['pic_ratio']
    )

    # 按百分制转换
    user_data['engagement_score'] = user_data['engagement_score'] * 100

    # 添加等级分类
    conditions2 = [
        user_data['engagement_score'] < 40,
        (user_data['engagement_score'] >= 40) & (user_data['engagement_score'] < 70),
        user_data['engagement_score'] >= 70
    ]
    choices2 = ['低参与度', '中等参与度', '高参与度']
    user_data['engagement_level'] = np.select(conditions2, choices2, default='-1')

    return user_data.sort_values('engagement_score', ascending=False)


# 4. 主函数
def user_engagement_model(df):
    # 计算基础指标
    user_stats = calculate_user_engagement(df)

    # 数据预处理
    preprocessed = preprocess_engagement_data(user_stats)

    # 计算参与度评分
    engagement_df = calculate_engagement_score(preprocessed)

    return engagement_df

# 运行参与度模型
engagement_results = user_engagement_model(moments)

# 查看结果
print(engagement_results[['uid', 'engagement_score', 'engagement_level']].head(10))

# 保存结果
engagement_results.to_csv("../tmp/user_engagement_scores.csv", index=False)
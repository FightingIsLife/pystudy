import pandas as pd


# 步骤1：加载原始数据
moments = pd.read_json('../big_json/moments/moments_merged.json', orient='records')
likes = pd.read_json('../big_json/likes/likes_merged.json', orient='records')

# 步骤2：计算用户历史平均点赞数
moments['avgLike'] = moments.groupby('uid')['likeCount'].transform('mean')


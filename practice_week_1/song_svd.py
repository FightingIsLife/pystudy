import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 练习一：数据加载与基本探索（Pandas基础）
# 1.读取数据
songs = pd.read_json('../aig_json/song.json')
printCols = ['name', 'artist', 'duration_minute','upload_year','popularity']

# 2. 数据概览
print(f"总歌曲数: {len(songs)}")
print(f"艺术家数量: {songs['artist'].nunique()}")
print(f"专辑数量: {songs['album'].nunique()}")

# 3. 统计缺失值
print("缺失值统计:")
print(songs.isnull().sum())

# 4. 计算歌曲平均时长（分钟） & 时间列转换
print(f"歌曲平均时长: {songs['duration'].mean() / 60_000:.2f}分钟")

songs['uploadTime'] = pd.to_datetime(songs['uploadTime'], unit='ms')  # 时间戳转换

# 练习二：数据清洗与特征工程（Pandas进阶）

# 1. 异常值处理：删除duration<10秒或>1小时的歌曲
safe_songs = songs[(songs.duration >= 10 * 1000) & (songs.duration < 3600 * 1000)].copy()

# 2. 创建新特征 duration转换为分钟 提取上传年份
safe_songs['duration_minute'] = safe_songs.duration / 60_000
safe_songs['upload_year'] = safe_songs.uploadTime.dt.year

# 3. 处理duration缺失值：用同专辑歌手的平均时长填充
album_avg_duration = safe_songs.groupby(['album','artist'])['duration'].transform('mean')
safe_songs['duration'] = safe_songs['duration'].fillna(album_avg_duration)

# 4. 分类转换：按收藏数分等级
safe_songs['popularity'] = pd.cut(safe_songs['collectionNum'],
                         bins=[-1, 0, 1000, 10000, float('inf')],
                         labels=['无人问津', '冷门', '热门', '爆款'])


# 练习三：分组分析与可视化（Pandas聚合）艺术家影响力分析（分组聚合）
# 1. 按艺术家分组统计,歌曲数量、平均时长、平均大小
artistGroup = safe_songs.groupby('artist').agg(song_count=('name', 'count'),avg_duration=('duration_minute', 'mean'),avg_size=('size', 'mean')).reset_index()


# 2. 找出作品最多的3位艺术家
top3 = artistGroup.sort_values('song_count',ascending=False).head(3)

# 3. 计算艺术家影响力得分 (歌曲数量 * 平均时长)
artistGroup['influence'] = artistGroup.song_count * artistGroup.avg_duration
top_influencers = artistGroup.sort_values('influence', ascending=False).head(5)
print("最具影响力艺术家:\n", top_influencers)

# 4. 艺术家歌曲时长分布箱线图
# safe_songs.boxplot(column='duration_minute', by='artist', figsize=(12, 6))
# plt.xticks(rotation=45)
# plt.title('不同艺术家歌曲时长分布')
# plt.suptitle('')  # 去除自动生成的标题
# plt.show()

# 练习四：音频特征分析（NumPy应用）探索音频文件的数值特征
# 1. 计算文件大小与时长的相关性
correlation = np.corrcoef(safe_songs['size'], safe_songs['duration'])[0, 1]
print(f"文件大小与时长的相关系数: {correlation:.3f}")

# 2. 创建音频质量指标 (文件大小 / 时长)
safe_songs['bitrate_kbps'] = (safe_songs['size'] * 8 / 1024) / (safe_songs['duration'] / 1000)  # kbps

# 3. 按质量指标分组
quality_bins = [0, 96, 192, 320, np.inf]
quality_labels = ['low', 'normal', 'high', 'top']

safe_songs['quality'] = pd.cut(safe_songs['bitrate_kbps'], bins=quality_bins, labels=quality_labels)

# 4. 各质量级别占比饼图
safe_songs['quality'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8))
plt.title('quality')
plt.show()

# 练习五：异常检测（高级分析）

# 1. 基于歌曲时长检测异常值 scipy.zscore

# outliers = songs[z_scores > 3]
# print(f"发现时长异常歌曲: {len(outliers)}首")
# print(outliers[['name', 'artist', 'duration_min']])

# 2. 检测文件大小与时长不匹配的歌曲
# predicted_size = songs['duration'] * songs['bitrate_kbps'].mean() / 8
# size_deviation = (songs['size'] - predicted_size) / songs['size']
# suspicious = songs[np.abs(size_deviation) > 0.5]
# print("\n文件大小异常的歌曲:")
# print(suspicious[['name', 'artist', 'size', 'duration_min']])

# 练习6：SVD应用 - 艺术家风格聚类

# 1. 创建艺术家特征矩阵  duration_min  bitrate_kbps upload_time
features = songs.groupby('artist').agg({
    'duration_min': ['mean', 'std'],
    'bitrate_kbps': 'mean',
    'upload_time': lambda x: (x.max() - x.min()).days  # 活跃时长
}).fillna(0)

# 特征重命名 'avg_duration', 'duration_variation', 'avg_quality', 'active_days'
features.columns = ['avg_duration', 'duration_variation', 'avg_quality', 'active_days']

# 2. 标准化特征矩阵

scaler = StandardScaler()
X = scaler.fit_transform(features)

# 3. SVD降维
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# 4. 保留前2个主成分进行可视化
X_2d = U[:, :2] @ np.diag(s[:2])

# 5. 聚类可视化
plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6)

# 标记艺术家名称
for i, artist in enumerate(features.index):
    plt.text(X_2d[i, 0], X_2d[i, 1], artist[:10], fontsize=9)

plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('艺术家风格聚类(SVD降维)')
plt.grid()
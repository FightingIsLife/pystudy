import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import datetime

moments = pd.read_json('../big_json/moments/moments_merged.json', orient='records')

user_avg = moments.groupby('uid')['likeCount'].mean().reset_index()
user_avg_dict = user_avg.set_index('uid')['likeCount'].to_dict()
global_avg = moments['likeCount'].mean()

#
MEAN = np.array([23.51766915184913, 0.6639661942559694, 10.476408329880353, 3.1730620289920752, 11.09243505849543])
STD = np.array([38.80781828501467, 0.472350645476996, 6.1347160317478595, 1.617235447564482, 3.2591848800595713])


# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 1. 必须与训练时相同的模型定义
class LikePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


# 2. 加载模型参数（重要：确保特征维度匹配）
model = LikePredictor(input_size=5).to(device)  # 关键修改1：将模型移到设备上

model.load_state_dict(torch.load('../big_json/pth/like_predictor.pth'))
model.eval()  # 切换为评估模式（关闭dropout等训练专用层）


# 3. 特征提取函数（必须与训练时一致）
def extract_features(dynamic):
    """从动态数据提取模型输入特征"""
    features = [
        len(dynamic['text']),  # 文本长度
        1 if dynamic['pics'] else 0,  # 是否含图片
        datetime.datetime.fromisoformat(dynamic['createTime']).hour,
        dynamic['type'],  # 动态类型
        # 发布者历史平均点赞数（需提前计算）
        user_avg_dict.get(dynamic['uid'], global_avg)
    ]
    print(features)
    return np.array(features, dtype=np.float32)

def standardize_features(features):
    print(features)
    print((features - MEAN) / STD)
    return np.array((features - MEAN) / STD, dtype=np.float32)

# 4. 预测函数
def predict_likes(dynamic):
    """预测单个动态的点赞数"""
    with torch.no_grad():  # 关闭梯度计算以提升性能
        # 1. 提取原始特征
        raw_features = extract_features(dynamic)

        # 2. 应用标准化（与训练时一致）
        standardized = standardize_features(raw_features)

        # 3. 转换到tensor并进行预测
        features_tensor = torch.tensor(standardized).unsqueeze(0)
        prediction = model(features_tensor)

        print(prediction)

        # 4. 反标准化输出（如果训练时对目标值做了标准化）
        # 但点赞数通常不标准化（改为log转换处理）
        return max(0, round(prediction.item()))


json_str = '''{
    "_id": 3772102012506,
    "uid": 3470256899,
    "type": 4,
    "location": {
        "area": "",
        "lng": 0,
        "lat": 0
    },
    "text": "夜景很美",
    "pics": [
        "https://oss.qingyujiaoyou.com/feed/and_3470256899_edf38bffca2fe5d99c321a080e44263d.jpg"
    ],
    "mentions": [],
    "status": 3,
    "createTime": "2025-06-28T14:38:57.562Z",
    "mentioned": false,
    "needPay": false,
    "visibleType": 0,
    "visibleUids": [],
    "ip": "36.24.251.252",
    "ipLocation": "浙江",
    "_class": "moment",
    "updateTime": "2025-06-28T14:39:38.685Z",
    "clickCount": 1,
    "likeCount": 10
}'''

# 转换为Python字典
data_dict = json.loads(json_str)
values = predict_likes(data_dict)
print(values)
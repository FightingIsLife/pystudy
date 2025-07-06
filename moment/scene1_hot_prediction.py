import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.preprocessing import StandardScaler


# 步骤1：加载原始数据
moments = pd.read_json('../big_json/moments/moments_merged.json', orient='records')

# 步骤2：计算用户历史平均点赞数
user_avg = pd.DataFrame({
    'uid': moments['uid'],
    'user_avg_likes': moments.groupby('uid')['likeCount'].transform('mean')
})


# 步骤3：特征工程
"""
* 动态文本长度（len(text)）
* 是否包含图片（pics是否为空）
* 发布时段（createTime的小时部分）
* 动态类型（type）
* 发布者历史平均点赞数（从历史数据计算）
"""


def extract_features(df):
    df = df.copy()

    df['createTime'] = pd.to_datetime(df['createTime'])
    df['hour'] = df['createTime'].dt.hour

    df['text_length'] = df['text'].str.len()
    df['has_image'] = df['pics'].str.len() > 0

    # 合并用户特征
    df = pd.merge(df, user_avg[['uid', 'user_avg_likes']], on='uid', how='left')
    df.fillna({'user_avg_likes': 0}, inplace=True)  # 新用户填充0

    return df[['text_length', 'has_image', 'hour', 'type', 'user_avg_likes', 'likeCount']]

# 步骤4：数据标准化, 为什么需要做标准化？
"""
简单比喻：
想象在比较身高(单位：米)和体重(单位：斤)：
    原始数据：身高1.7米 vs 体重120斤
    标准化后：身高0.3 vs 体重0.2（假设值）
这样就不会因为体重数字更大而被认为更重要
"""
feature_df = extract_features(moments)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_df.drop('likeCount', axis=1))


# PyTorch数据集实现
class MomentDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# 划分数据集
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    scaled_features,
    feature_df['likeCount'],
    test_size=0.2,
    random_state=42
)

train_dataset = MomentDataset(X_train, y_train)
test_dataset = MomentDataset(X_test, y_test)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


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


# 初始化模型并移到GPU
model = LikePredictor(input_size=5).to(device)  # 关键修改1：将模型移到设备上

# 训练配置
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(20):
    model.train()
    for inputs, targets in train_loader:
        # 将数据移到GPU
        inputs = inputs.to(device)  # 关键修改2：输入数据移到GPU
        targets = targets.to(device)  # 关键修改3：目标数据移到GPU

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # 验证集评估
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for inputs, targets in test_loader:
            inputs = inputs.to(device)  # 验证数据同样需要移到GPU
            targets = targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()

        print(f'Epoch {epoch + 1} | Test MSE: {test_loss / len(test_loader):.4f}')

# 保存模型（指定保存到CPU以便通用加载）
model.to('cpu')
torch.save(model.state_dict(), '../big_json/pth/like_predictor.pth')
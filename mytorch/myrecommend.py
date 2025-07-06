import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ========== 1. 数据处理层 ==========
# 类似Java的DAO层+DTO转换
class UserBehaviorDataset(Dataset):
    """用户行为数据集 (类似Spring的Repository)"""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_user_data(file_path):
    """加载用户数据 (类似Spring的Service层方法)"""
    # 假设CSV包含: user_id, age, gender, usage_duration, likes, shares, post_category
    df = pd.read_csv(file_path)

    # 特征工程（类似业务逻辑处理）
    df['interaction_rate'] = (df['likes'] + df['shares']) / df['usage_duration']

    # 拆分特征和标签
    X = df[['age', 'interaction_rate']].values  # 实际中可加入更多特征
    y = df['post_category'].values  # 预测用户喜欢的帖子类别

    return X, y


# ========== 2. 模型层 ==========
# 类似Spring的Controller + Service
class RecommendationModel(nn.Module):
    """个性化推荐模型 (类似@RestController)"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # 用户特征提取层
        self.layer2 = nn.Linear(hidden_size, output_size)  # 类别预测层
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x):
        """预测流程 (类似@RequestMapping)"""
        x = torch.relu(self.layer1(x))  # 特征转换（类似业务逻辑）
        x = self.dropout(x)  # 随机失活（类似校验过滤）
        x = self.layer2(x)  # 输出预测
        return x


# ========== 3. 训练服务 ==========
# 类似Spring Boot的Application启动类
def train_recommendation_model():
    # 加载数据（可替换为你的社交APP数据）
    X, y = load_user_data('user_behavior.csv')  # 示例文件名
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 创建数据集（类似初始化DataSource）
    train_dataset = UserBehaviorDataset(X_train, y_train)
    test_dataset = UserBehaviorDataset(X_test, y_test)

    # 参数配置（类似application.properties）
    BATCH_SIZE = 32
    EPOCHS = 50
    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = len(np.unique(y))  # 分类数量

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 初始化模型（类似Bean注入）
    model = RecommendationModel(INPUT_SIZE, 128, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()  # 损失函数（类似校验规则）
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

    # 训练循环（类似服务主循环）
    train_losses = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for features, labels in train_loader:
            # 正向传播（类似处理请求）
            outputs = model(features)
            loss = criterion(outputs, labels.long())

            # 反向传播（类似异常处理/状态回滚）
            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}')

    # 保存模型（类似持久化）
    torch.save(model.state_dict(), 'recommendation_model.pth')

    # 可视化训练过程（类似监控端点）
    plt.plot(train_losses)
    plt.title('Training Loss (类似性能监控)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')

    return model


# ========== 4. 预测服务 ==========
# 类似Spring的REST API端点
class PredictionService:
    def __init__(self, model_path):
        # 加载训练好的模型（类似@PostConstruct初始化）
        self.model = RecommendationModel(input_size=2, hidden_size=128, output_size=10)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # 设置为评估模式

    def predict_preference(self, user_features):
        """预测用户偏好 (类似@PostMapping("/predict"))"""
        with torch.no_grad():
            features = torch.tensor(user_features, dtype=torch.float32).unsqueeze(0)
            output = self.model(features)
            _, predicted = torch.max(output, 1)
            return predicted.item()  # 返回预测的类别ID


# ========== 5. 主程序 ==========
if __name__ == "__main__":
    # 训练模型（首次运行时执行）
    # trained_model = train_recommendation_model()

    # 启动预测服务（类似SpringApplication.run())
    predictor = PredictionService('recommendation_model.pth')

    # 模拟用户特征：[年龄, 互动率]
    test_user = [28, 0.65]  # 28岁，互动率65%

    # 进行预测（类似API调用）
    preferred_category = predictor.predict_preference(test_user)
    print(f"用户可能喜欢的帖子类别: {preferred_category}")
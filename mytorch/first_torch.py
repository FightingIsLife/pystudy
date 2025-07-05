import torch
import torch.nn as nn
import torch.optim as optim

# 1. 数据准备（暂时当成黑盒子）
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # XOR运算

# 2. 定义一个最简单的神经网络（不懂细节没关系！）
model = nn.Sequential(
     nn.Linear(2, 2),  # 输入2个数字，输出2个特征
     nn.ReLU(),        # 激活函数（暂时理解为“开关”）
     nn.Linear(2, 1)   # 输出最终结果
)

# 3. 配置训练工具（固定套路）
criterion = nn.MSELoss()  # 损失计算器
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 优化器

# 4. 开始训练（看输入输出变化即可）
for epoch in range(1000):
     y_pred = model(X)         # 前向计算
     loss = criterion(y_pred, y) 
     optimizer.zero_grad()     # 清空梯度
     loss.backward()           # 反向传播（自动计算梯度）
     optimizer.step()          # 更新参数

     if epoch % 100 == 0: 
         print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 5. 测试效果！
test_data = torch.tensor([[0., 0.], [0.5, 0.5]])
print(f"预测结果：{model(test_data).detach()}")
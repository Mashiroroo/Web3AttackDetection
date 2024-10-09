import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# 检查并使用GPU
gpu_id = 1  # 根据你的需求选择ID
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据集类
class TransactionDataset(Dataset):
    def __init__(self, folder_path, label):
        self.data = []
        self.label = label
        self.vectorizer = CountVectorizer(max_features=20000)

        # 加载数据
        self.load_data(folder_path)

    def load_data(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(folder_path, filename)) as f:
                        data = json.load(f)
                        features = {
                            'balance_change': str(data['balance_change'].get('balanceChanges', [])),
                            'basic_info': str(data['profile'].get('basic_info', {})),
                            'fund_flow': str(data['profile'].get('fund_flow', [])),
                            'token_infos': str(data['profile'].get('token_infos', [])),
                            'address_label': str(data['address_label'].get('labels', [])),
                            'account_labels': str(data['trace'].get('accountLabels', 0)),
                            'data_map': str(data['trace'].get('dataMap', {})),
                            'gas_flame': str(data['trace'].get('gasFlame', [])),
                            'main_trace': str(data['trace'].get('mainTrace', [])),
                            'main_trace_node_count': str(data['trace'].get('mainTraceNodeCount', 0)),
                            'parent_id_map': str(data['trace'].get('parentIdMap', {})),
                            'tidy_trace': str(data['trace'].get('tidyTrace', [])),
                            'tidy_trace_node_count': str(data['trace'].get('tidyTraceNodeCount', 0)),
                            'state_change': str(data['state_change'].get('stateChanges', []))
                        }
                        self.data.append(features)
                except Exception as e:
                    print(f"读取 {filename} 时出错: {e}")

    def vectorize(self):
        texts = [str(item) for item in self.data]
        self.vectorized_features = self.vectorizer.fit_transform(texts).toarray()

        # 数据标准化
        scaler = StandardScaler()
        self.vectorized_features = scaler.fit_transform(self.vectorized_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.vectorized_features[idx], dtype=torch.float32), torch.tensor(self.label,
                                                                                              dtype=torch.float32)


# Transformer 模型构建
class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads, num_classes, dim_feedforward, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)  # 输入特征嵌入
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=n_heads, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.fc = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 增加序列维度
        x = x.unsqueeze(1)  # 形状为 [batch_size, seq_len=1, feature_dim]
        x = self.embedding(x)  # 形状为 [batch_size, seq_len=1, embedding_dim]
        x = self.transformer_encoder(x)  # [batch_size, seq_len=1, embedding_dim]
        x = x.mean(dim=1)  # 取平均，形状为 [batch_size, embedding_dim]
        x = self.fc(x)
        return self.sigmoid(x)


# 训练模型函数
def train_model(attack_folder, non_attack_folder, batch_size, epochs):
    attack_dataset = TransactionDataset(attack_folder, label=1)
    non_attack_dataset = TransactionDataset(non_attack_folder, label=0)

    attack_dataset.vectorize()
    non_attack_dataset.vectorize()

    train_dataset = torch.utils.data.ConcatDataset([attack_dataset, non_attack_dataset])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化 Transformer 模型
    model = TransformerModel(input_dim=20000, n_heads=4, num_classes=1, dim_feedforward=256, num_layers=2).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)  # 学习率调度器

    history_list = []
    best_val_loss = float('inf')  # 用于保存最佳验证损失

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()  # 设置模型为训练模式
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs).squeeze()  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)
        print(f"Epoch Loss: {average_loss:.4f}")
        history_list.append(average_loss)

        # 更新学习率调度器
        scheduler.step(average_loss)

        # 可选：实现早停法
        if average_loss < best_val_loss:
            best_val_loss = average_loss
            torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型

    return model, history_list


# 定义数据批量大小
batch_size = 64

# 训练模型
model, history_list = train_model('../data', '../normal_data', batch_size, epochs=80)

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))


# 预测
def predict(model, test_data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(test_data.vectorized_features, dtype=torch.float32).to(device)
        outputs = model(inputs).squeeze()
        predictions = (outputs.cpu().numpy() > 0.5).astype(int)
    return predictions


# 加载测试数据
test_data = TransactionDataset('../data', label=1)
test_data.vectorize()
non_attack_test_data = TransactionDataset('../normal_data', label=0)
non_attack_test_data.vectorize()

# 合并测试数据
test_features = np.concatenate((test_data.vectorized_features, non_attack_test_data.vectorized_features))
test_labels = np.concatenate((np.ones(len(test_data)), np.zeros(len(non_attack_test_data))))

# 预测
predictions = predict(model, test_data)

# 输出预测结果
print(predictions)


# 可视化训练过程曲线
def plot_training_curves(history_list):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history_list) + 1), history_list, label='训练损失')
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    # plt.show()


# 绘制训练过程曲线
plot_training_curves(history_list)

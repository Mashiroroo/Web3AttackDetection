import json
import os

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# 检查并使用GPU
gpu_id = 1  # 根据你的需求选择ID
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据集类
class TransactionDataset(Dataset):
    def __init__(self, folder_path):
        self.data = []
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
        return torch.tensor(self.vectorized_features[idx], dtype=torch.float32)


# Transformer 模型构建
class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads, num_classes, dim_feedforward, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 256)  # 增加嵌入维度
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=n_heads, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.fc = nn.Linear(256, num_classes)  # 更新全连接层的输入维度
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # 形状为 [batch_size, seq_len=1, feature_dim]
        x = self.embedding(x)  # 形状为 [batch_size, seq_len=1, embedding_dim]
        x = self.transformer_encoder(x)  # [batch_size, seq_len=1, embedding_dim]
        x = x.mean(dim=1)  # 取平均，形状为 [batch_size, embedding_dim]
        x = self.fc(x)
        return self.sigmoid(x)


# 预测函数
def predict(model, dataset):
    model.eval()
    probabilities = []

    with torch.no_grad():
        inputs = torch.tensor(dataset.vectorized_features, dtype=torch.float32).to(device)
        outputs = model(inputs).squeeze()  # 预测
        probabilities = outputs.cpu().numpy()  # 将预测结果转为numpy数组

    return probabilities


# 加载最佳模型
model = TransformerModel(input_dim=20000, n_heads=4, num_classes=1, dim_feedforward=512, num_layers=4).to(device)
model.load_state_dict(torch.load('best_model.pth'))

# 加载测试数据
test_folder = '../normal_data'  # 修改为你要遍历的文件夹路径
test_dataset = TransactionDataset(test_folder)
test_dataset.vectorize()  # 向量化数据

# 预测
probabilities = predict(model, test_dataset)

# 输出每个数据的攻击交易概率
for idx, prob in enumerate(probabilities):
    print(f"样本 {idx + 1} 的攻击交易概率: {prob:.4f}")

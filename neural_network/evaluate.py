import json
import os
import numpy as np
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
    def __init__(self, folder_path, label):
        self.data = []
        self.labels = []
        self.file_names = []  # 存储文件名
        self.label = label
        self.vectorizer = CountVectorizer(max_features=20000)
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
                        self.labels.append(self.label)
                        self.file_names.append(filename)  # 存储文件名
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
        return (torch.tensor(self.vectorized_features[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32),
                self.file_names[idx])  # 返回文件名


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
        x = x.unsqueeze(1)  # 增加序列维度
        x = self.embedding(x)  # 嵌入
        x = self.transformer_encoder(x)  # 编码
        x = x.mean(dim=1)  # 取平均
        x = self.fc(x)
        return self.sigmoid(x)


# 加载模型和数据并进行预测
def load_model_and_predict(model_path, test_data_folder):
    # 加载模型
    input_dim = 20000
    model = TransformerModel(input_dim=input_dim, n_heads=4, num_classes=1, dim_feedforward=256, num_layers=2).to(
        device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载测试数据
    test_data = TransactionDataset(test_data_folder, label=1)
    test_data.vectorize()

    # 进行预测
    with torch.no_grad():
        inputs = torch.tensor(test_data.vectorized_features, dtype=torch.float32).to(device)
        outputs = model(inputs).squeeze()
        probabilities = outputs.cpu().numpy()

    return test_data.file_names, probabilities


# 主程序
if __name__ == "__main__":
    model_path = 'best_model.pth'  # 替换为你的模型路径
    test_data_folder = '../data'  # 替换为你的测试数据文件夹路径

    file_names, probabilities = load_model_and_predict(model_path, test_data_folder)

    # 输出结果
    for file_name, probability in zip(file_names, probabilities):
        print(f"文件: {file_name}, 攻击概率: {probability:.4f}")

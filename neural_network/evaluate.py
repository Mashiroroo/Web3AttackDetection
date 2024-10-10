import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# 检查并使用GPU
gpu_id = 1  # 根据你的需求选择ID
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据集类
class TransactionDataset:
    def __init__(self, folder_path):
        self.data = []
        self.vectorizer = None
        self.scaler = None

        # 加载数据
        self.load_data(folder_path)

    def load_data(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                try:
                    file_path = os.path.join(folder_path, filename)
                    file_size = os.path.getsize(file_path)  # 获取文件大小
                    with open(file_path) as f:
                        data = json.load(f)
                        features = {
                            'balance_change': data['balance_change'].get('balanceChanges', []),
                            'basic_info': data['profile'].get('basic_info', {}),
                            'fund_flow': data['profile'].get('fund_flow', []),
                            'token_infos': data['profile'].get('token_infos', []),
                            'address_label': data['address_label'].get('labels', []),
                            'account_labels': data['trace'].get('accountLabels', []),
                            'data_map': data['trace'].get('dataMap', {}),
                            'gas_flame': data['trace'].get('gasFlame', []),
                            'main_trace': data['trace'].get('mainTrace', []),
                            'main_trace_node_count': data['trace'].get('mainTraceNodeCount', 0),
                            'parent_id_map': data['trace'].get('parentIdMap', {}),
                            'tidy_trace': data['trace'].get('tidyTrace', []),
                            'tidy_trace_node_count': data['trace'].get('tidyTraceNodeCount', 0),
                            'state_change': data['state_change'].get('stateChanges', []),
                            'file_size': file_size  # 添加文件大小特征
                        }
                        self.data.append(features)
                except Exception as e:
                    print(f"读取 {filename} 时出错: {e}")

    def vectorize(self):
        texts = [str(item) for item in self.data]
        vectorizer = CountVectorizer(max_features=20000)
        self.vectorized_features = vectorizer.fit_transform(texts).toarray()

        # 对文件大小进行处理
        file_sizes = [item['file_size'] for item in self.data]
        file_sizes = np.log1p(file_sizes)  # 对数变换，避免负数
        file_sizes = file_sizes.reshape(-1, 1)  # 调整维度以便拼接

        # 拼接向量化特征和文件大小特征
        self.vectorized_features = np.hstack((self.vectorized_features, file_sizes))

        self.scaler = StandardScaler()
        self.vectorized_features = self.scaler.fit_transform(self.vectorized_features)

    def get_vectorized_data(self):
        return torch.tensor(self.vectorized_features, dtype=torch.float32)


# Transformer 模型构建
class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads, num_classes, dim_feedforward, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 256)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=n_heads, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.batch_norm = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return self.sigmoid(self.fc(x))


# 评估函数
def evaluate_model(model, dataset):
    model.eval()
    inputs = dataset.get_vectorized_data().to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.cpu().numpy()


# 主程序
def main(model_path, vectorizer_path, scaler_path, data_folder):
    # 加载模型
    input_dim = 20000 + 1  # 20000是特征数量，1是文件大小特征
    model = TransformerModel(input_dim=input_dim, n_heads=4, num_classes=1, dim_feedforward=512, num_layers=6)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # 加载向量化器和标准化器
    with open(vectorizer_path, 'rb') as v_file:
        vectorizer = pickle.load(v_file)

    with open(scaler_path, 'rb') as s_file:
        scaler = pickle.load(s_file)

    # 创建数据集
    dataset = TransactionDataset(data_folder)
    dataset.vectorize()

    # 评估模型
    predictions = evaluate_model(model, dataset)
    predictions_percentage = (predictions > 0.5).astype(int)

    # 输出结果
    for filename, prediction in zip(os.listdir(data_folder), predictions_percentage):
        print(f"文件: {filename}, 预测: {prediction[0] * 100:.2f}%")


if __name__ == "__main__":
    model_path = 'transformer_model.pth'
    vectorizer_path = 'vectorizer.pkl'
    scaler_path = 'scaler.pkl'
    data_folder = '../test_data'  # 指定数据文件夹
    main(model_path, vectorizer_path, scaler_path, data_folder)

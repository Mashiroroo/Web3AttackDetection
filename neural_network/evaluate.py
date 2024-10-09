import json
import os

import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# 检查并使用GPU
gpu_id = 1  # 根据你的需求选择ID
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义数据处理类
class TransactionDataset:
    def __init__(self, folder_path):
        self.data = []
        self.vectorizer = CountVectorizer(max_features=20000)
        self.scaler = StandardScaler()

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
        self.vectorized_features = self.scaler.fit_transform(self.vectorized_features)

    def get_features(self):
        return torch.tensor(self.vectorized_features, dtype=torch.float32).to(device)


# 定义简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(20000, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


# 评测模型函数
def evaluate_model(model, folder_path):
    # 加载数据并处理
    dataset = TransactionDataset(folder_path)
    dataset.vectorize()

    # 获取特征并预测
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        features = dataset.get_features()
        outputs = model(features).squeeze()

        # 转换为概率
        probabilities = outputs.cpu().numpy()

    return probabilities


# 加载训练好的模型
def load_trained_model(model_path):
    model = SimpleModel().to(device)
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    # 加载模型
    model_path = 'best_model.pth'
    model = load_trained_model(model_path)

    # 选择要评测的数据文件夹
    folder_path = input("请输入要评测的数据文件夹路径：")

    # 评测数据并输出每条交易为攻击的概率
    probabilities = evaluate_model(model, folder_path)

    # 输出概率
    for i, prob in enumerate(probabilities):
        print(f"Transaction {i + 1}: Attack Probability: {prob:.4f}")

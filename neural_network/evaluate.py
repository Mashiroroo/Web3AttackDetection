import json
import os
import pickle

import torch
import torch.nn as nn

# 检查并使用GPU
gpu_id = 1  # 根据你的需求选择ID
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


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


# 加载模型
def load_model(model_path):
    model = TransformerModel(input_dim=20000, n_heads=4, num_classes=1, dim_feedforward=512, num_layers=6).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 加载向量化器和标准化器
def load_vectorizer_and_scaler(vectorizer_path, scaler_path):
    with open(vectorizer_path, 'rb') as v_file:
        vectorizer = pickle.load(v_file)
    with open(scaler_path, 'rb') as s_file:
        scaler = pickle.load(s_file)
    return vectorizer, scaler


# 处理数据并进行预测
def evaluate_on_folder(model, vectorizer, scaler, folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(folder_path, filename)) as f:
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
                    }
                    # 向量化
                    text = str(features)
                    vectorized_features = vectorizer.transform([text]).toarray()
                    scaled_features = scaler.transform(vectorized_features)

                    # 转为张量
                    input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

                    # 预测
                    with torch.no_grad():
                        output = model(input_tensor).squeeze()
                        prediction = (output > 0.5).cpu().numpy()

                    results.append({'filename': filename, 'prediction': prediction.item()})
            except Exception as e:
                print(f"读取 {filename} 时出错: {e}")

    return results


# 设置参数并执行评估
model_path = 'best_model.pth'  # 模型文件路径
vectorizer_path = 'vectorizer.pkl'
scaler_path = 'scaler.pkl'
test_folder = '../test_data'  # 测试数据文件夹路径

model = load_model(model_path)
vectorizer, scaler = load_vectorizer_and_scaler(vectorizer_path, scaler_path)
results = evaluate_on_folder(model, vectorizer, scaler, test_folder)

# 输出预测结果
for result in results:
    print(f"Filename: {result['filename']}, Prediction: {result['prediction']}")

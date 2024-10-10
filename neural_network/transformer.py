import json
import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

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
                        self.data.append(features)
                except Exception as e:
                    print(f"读取 {filename} 时出错: {e}")

    def vectorize(self):
        texts = [str(item) for item in self.data]
        self.vectorized_features = self.vectorizer.fit_transform(texts).toarray()
        self.vectorized_features = self.scaler.fit_transform(self.vectorized_features)

    def save_vectorizer_and_scaler(self, vectorizer_path, scaler_path):
        with open(vectorizer_path, 'wb') as v_file:
            pickle.dump(self.vectorizer, v_file)
        with open(scaler_path, 'wb') as s_file:
            pickle.dump(self.scaler, s_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.vectorized_features[idx], dtype=torch.float32), torch.tensor(self.label,
                                                                                              dtype=torch.float32)


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


# 模型评估
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs).squeeze()
            val_loss += criterion(val_outputs, val_labels).item()
            all_labels.extend(val_labels.cpu().numpy())
            all_outputs.extend((val_outputs > 0.5).cpu().numpy())

    average_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_outputs)
    f1 = f1_score(all_labels, all_outputs)

    return average_val_loss, accuracy, f1


# 训练模型函数
def train_model(attack_folder, non_attack_folder, batch_size, epochs):
    attack_dataset = TransactionDataset(attack_folder, label=1)
    non_attack_dataset = TransactionDataset(non_attack_folder, label=0)

    all_data = attack_dataset.data + non_attack_dataset.data
    vectorizer = CountVectorizer(max_features=20000)
    scaler = StandardScaler()

    vectorized_features = vectorizer.fit_transform([str(item) for item in all_data]).toarray()
    vectorized_features = scaler.fit_transform(vectorized_features)

    attack_len = len(attack_dataset)
    attack_dataset.vectorized_features = vectorized_features[:attack_len]
    non_attack_dataset.vectorized_features = vectorized_features[attack_len:]

    train_dataset, val_dataset = train_test_split(
        torch.utils.data.ConcatDataset([attack_dataset, non_attack_dataset]),
        test_size=0.2,
        random_state=42
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerModel(input_dim=20000, n_heads=4, num_classes=1, dim_feedforward=512, num_layers=6).to(device)

    # 使用加权损失函数
    attack_weight = 10  # 可调整
    non_attack_weight = 1
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([attack_weight]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    history_list = []
    val_history_list = []
    accuracy_list = []
    f1_list = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)
        print(f"Epoch Loss: {average_loss:.4f}")
        history_list.append(average_loss)

        scheduler.step(average_loss)

        average_val_loss, accuracy, f1 = evaluate_model(model, val_loader, criterion)

        print(f"Validation Loss: {average_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        val_history_list.append(average_val_loss)
        accuracy_list.append(accuracy)
        f1_list.append(f1)

        # if average_val_loss < best_val_loss:
        #     best_val_loss = average_val_loss
        #     torch.save(model.state_dict(), 'best_model.pth')
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= 5:
        #         print("Early stopping...")
        #         break

    with open('vectorizer.pkl', 'wb') as v_file:
        pickle.dump(vectorizer, v_file)
    with open('scaler.pkl', 'wb') as s_file:
        pickle.dump(scaler, s_file)

    return model, history_list, val_history_list, accuracy_list, f1_list

# 可视化训练过程曲线
def plot_training_curves(history_list, val_history_list, accuracy_list, f1_list):
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(history_list) + 1), history_list, label='train loss', color='blue')
    plt.plot(range(1, len(val_history_list) + 1), val_history_list, label='test loss', color='orange')
    plt.title('train&test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, label='accuracy', color='green')
    plt.title('accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制F1分数曲线
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(f1_list) + 1), f1_list, label='F1', color='red')
    plt.title('F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    # 保存图形
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前图形


# 设置参数并开始训练
attack_folder = '../data'
non_attack_folder = '../normal_data'
batch_size = 64
epochs = 50

model, history_list, val_history_list, accuracy_list, f1_list = train_model(attack_folder, non_attack_folder, batch_size, epochs)

# 可视化训练过程曲线
plot_training_curves(history_list, val_history_list, accuracy_list, f1_list)



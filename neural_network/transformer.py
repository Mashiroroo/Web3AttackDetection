import json
import os

import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, TextVectorization, Embedding, GlobalAveragePooling1D


# 生成器函数，逐步读取JSON文件，并批量处理数据
def data_generator(folder_path, label, batch_size):
    batch_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename)) as f:
                data = json.load(f)
                features = {
                    'label': label,
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
                batch_data.append(features)
                if len(batch_data) == batch_size:
                    yield pd.DataFrame(batch_data)
                    batch_data = []
    if batch_data:
        yield pd.DataFrame(batch_data)


# TextVectorization将字符串特征转为词向量
def process_string_features(df, vectorizer):
    string_features = df.drop(columns=['label'])
    vectorized_features = []
    for col in string_features.columns:
        vectorized = vectorizer(string_features[col].values)
        vectorized_features.append(vectorized.numpy())  # 确保转换为numpy数组
    return np.concatenate(vectorized_features, axis=1), df['label']  # 合并所有特征


# 批量加载数据和训练模型
def train_in_batches(attack_folder, non_attack_folder, batch_size, vectorizer, model, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # 处理攻击样本
        for attack_batch in data_generator(attack_folder, 1, batch_size):
            X_attack, y_attack = process_string_features(attack_batch, vectorizer)
            model.fit(X_attack, np.array(y_attack), epochs=1, batch_size=batch_size, verbose=1)

        # 处理非攻击样本
        for non_attack_batch in data_generator(non_attack_folder, 0, batch_size):
            X_non_attack, y_non_attack = process_string_features(non_attack_batch, vectorizer)
            model.fit(X_non_attack, np.array(y_non_attack), epochs=1, batch_size=batch_size, verbose=1)


# 加载数据的函数
def load_data(folder_path, label):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename)) as f:
                data = json.load(f)
                features = {
                    'label': label,
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
                data_list.append(features)
    return pd.DataFrame(data_list)


# 定义数据批量大小
batch_size = 32

# 初始化TextVectorization层
vectorizer = TextVectorization(output_mode='int', max_tokens=20000)

# 初始化并适配词向量
sample_data = next(data_generator('../blocksec/data', 1, batch_size=1))  # 加载一个样本来适配
vectorizer.adapt(sample_data.drop(columns=['label']).values.flatten())

# Transformer 模型构建
input_layer = Input(shape=(None,), dtype='int64')  # 输入层的形状根据文本长度自动调整

# 使用Embedding层将词索引转换为词向量
embedding_layer = Embedding(input_dim=20000, output_dim=128)(input_layer)

# 使用GlobalAveragePooling1D来处理变长序列
x = GlobalAveragePooling1D()(embedding_layer)

# 继续构建模型
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
train_in_batches('../blocksec/data', '../blocksec/normal_data', batch_size, vectorizer, model, epochs=10)

# 在训练后划分数据集以进行测试
test_data = pd.concat([load_data('../blocksec/data', 1), load_data('../blocksec/normal_data', 0)], ignore_index=True)
X_test, y_test = process_string_features(test_data, vectorizer)

# 预测
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)

# 输出预测结果
print(predictions)

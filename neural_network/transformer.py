import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, \
    GlobalAveragePooling1D


# 读取JSON文件的函数
def load_data_from_folder(folder_path, label):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename)) as f:
                data = json.load(f)
                features = {
                    'balance_change': data['balance_change'].get('balanceChanges', 0),
                    'basic_info': data['profile'].get('basic_info', 0),
                    'fund_flow': data['profile'].get('fund_flow', 0),
                    'token_infos': data['profile'].get('token_infos', 0),
                    'address_label': data['address_label'].get('labels', ''),
                    'account_labels': data['trace'].get('accountLabels', 0),
                    'data_map': data['trace'].get('dataMap', 0),
                    'gas_flame': data['trace'].get('gasFlame', 0),
                    'main_trace': data['trace'].get('mainTrace', 0),
                    'main_trace_node_count': data['trace'].get('mainTraceNodeCount', 0),
                    'parent_id_map': data['trace'].get('parentIdMap', 0),
                    'tidy_trace': data['trace'].get('tidyTrace', 0),
                    'tidy_trace_node_count': data['trace'].get('tidyTraceNodeCount', 0),
                    'state_change': data['state_change'].get('stateChanges', 0),
                }
                data_list.append(features)
    return pd.DataFrame(data_list)


# 加载数据
attack_data = load_data_from_folder('../blocksec/data', 1)
non_attack_data = load_data_from_folder('../blocksec/normal_data', 0)

# 合并数据
data = pd.concat([attack_data, non_attack_data], ignore_index=True)

# 特征和标签
features = data[['max_depth', 'max_repeat', 'max_amount', 'function_call_chain', 'deploy_time', 'call_interval']]
labels = data['is_malicious']

# 数据预处理
X = np.array(features)
y = np.array(labels)

# 训练/测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Transformer 模型构建
input_layer = Input(shape=(X_train.shape[1],))


# Transformer 块
def transformer_block(inputs, num_heads=4, ff_dim=32, dropout_rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


x = transformer_block(input_layer)
x = GlobalAveragePooling1D()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 预测
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)

# 输出预测结果
print(predictions)

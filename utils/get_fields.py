import csv
import os


def get_tx_hash(tx):
    return tx[-66:]


# 从数据集中读取包含chain 和 包含tx 的两个列表，他们的序列是一一对应的
def get_chain_and_tx():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '../data/all_in_one.csv')
    tx_list = []
    chain_list = []
    # print(os.getcwd())
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
        for row in rows:
            tx = row[6]
            chain = row[2]
            if '|' in tx:
                temp_tx_list = tx.split('|')
                tx_list = tx_list + temp_tx_list
            else:
                tx_list.append(tx)
            temp_chain_list = chain.split(' ')
            chain_list = chain_list + temp_chain_list
            tx_list = [get_tx_hash(tx) for tx in tx_list]
    return chain_list, tx_list


# 根据tx，寻找他对应的chain
def get_chain_by_tx(tx):
    chain_list, tx_list = get_chain_and_tx()
    # print(tx_list)
    if tx in tx_list:
        index = tx_list.index(tx)
        return chain_list[index]
    else:
        return None


if __name__ == '__main__':
    chain_list, tx_list = get_chain_and_tx()
    print(chain_list)
    print(tx_list)

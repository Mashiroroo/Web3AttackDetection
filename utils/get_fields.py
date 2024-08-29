import csv
import os


def get_tx_hash(tx):
    return tx[-66:]


def get_chain_and_tx(input_file):
    tx_list = []
    chain_list = []
    # print(os.getcwd())
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # next(reader)
        rows = list(reader)
        for row in rows:
            tx = row[-4]
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


def get_chain_by_tx(tx):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '../dataset/utf8Format_standard.csv')
    chain_list, tx_list = get_chain_and_tx(file_path)
    # print(tx_list)
    if tx in tx_list:
        index = tx_list.index(tx)
        return chain_list[index]
    else:
        return None


if __name__ == '__main__':
    print(get_chain_and_tx('../dataset/utf8Format_standard.csv'))

import csv
import os


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
    return chain_list, tx_list


if __name__ == '__main__':
    print(get_chain_and_tx())

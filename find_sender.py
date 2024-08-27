import csv

input_file = r'./transaction_data_utf8.csv'
tx_list = []
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    rows = list(reader)
    for row in rows:
        tx = row[-4]
        if '|' in tx:
            temp_list = tx.split('|')
            tx_list = tx_list + temp_list
        else:
            tx_list.append(tx)

unique_tx_list = set(tx_list)


import csv

from web3 import Web3

from utils.query import Query
from utils.get_tx_hash import get_tx_hash

input_file = r'data/utf8Format_standard.csv'
tx_list = []
chain_list = []
query = Query(rpc_node=None, transaction=None, chain=None)

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

sender_list = []
for chain, tx in zip(chain_list, tx_list):
    # print(chain, tx)
    query.chain = chain
    query.transaction = get_tx_hash(tx)
    query.rpc_node = query.rpc_node_dist.get(chain, None)
    if query.rpc_node:
        query.w3 = Web3(Web3.HTTPProvider(query.rpc_node))
    sender = query.find_sender()
    sender_list.append(sender)
print(sender_list)
print(len(sender_list))




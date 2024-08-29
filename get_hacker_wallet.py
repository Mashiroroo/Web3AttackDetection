from web3 import Web3
from utils.get_fields import get_chain_and_tx
from utils.processor import Processor
from utils.get_tx_hash import get_tx_hash

query = Processor(rpc_node=None, transaction=None, chain=None, config_path='config.yaml')
chain_list, tx_list = get_chain_and_tx(input_file=r'dataset/utf8Format_standard.csv')

sender_list = []
for chain, tx in zip(chain_list, tx_list):
    # print(chain, tx)
    query.chain = chain
    query.transaction = get_tx_hash(tx)
    query.rpc_node = query.config['rpc_nodes'].get(chain, None)
    if query.rpc_node:
        query.w3 = Web3(Web3.HTTPProvider(query.rpc_node))
    sender = query.find_sender()
    sender_list.append(sender)
print(sender_list)
print(len(sender_list))

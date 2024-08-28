import csv
from web3 import Web3
from utils.get_fields import get_chain_and_tx
from utils.processor import Processor
from utils.get_tx_hash import get_tx_hash

chain_list, tx_list = get_chain_and_tx(input_file=r'dataset/utf8Format_standard.csv')
loader = Processor(rpc_node=None, transaction=None, chain=None, config_path='config.yaml')

for chain, tx in zip(chain_list, tx_list):
    # print(chain, tx)
    loader.chain = chain
    loader.transaction = get_tx_hash(tx)
    loader.rpc_node = loader.config['rpc_nodes'].get(chain, None)
    if loader.rpc_node:
        loader.w3 = Web3(Web3.HTTPProvider(loader.rpc_node))
    loader.load_transaction(output_dir='data')

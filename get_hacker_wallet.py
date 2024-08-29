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

filtered_addresses = [address for address in sender_list if address is not None]
# 使用 set 去重
unique_senders = set(filtered_addresses)

with open('./hacker_wallet/hacker_wallet.txt', 'w') as file:
    for sender in unique_senders:
        file.write(f"{sender}\n")

print(f"去重后的发送者数量: {len(unique_senders)}")

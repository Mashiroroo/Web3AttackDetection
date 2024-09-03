from web3 import Web3
from utils.get_fields import get_chain_and_tx
from utils.processor import Processor

processor = Processor(transaction=None, chain=None)
chain_list, tx_list = get_chain_and_tx()

sender_list = []
for chain, tx in zip(chain_list, tx_list):
    # print(chain, tx)
    processor.chain = chain
    processor.transaction = tx
    processor.rpc_node = processor.config['rpc_nodes'].get(chain, None)
    if processor.rpc_node:
        processor.w3 = Web3(Web3.HTTPProvider(processor.rpc_node))
    sender = processor.find_sender()
    sender_list.append(sender)
print(sender_list)

filtered_addresses = [address for address in sender_list if address is not None]
# 使用 set 去重
unique_senders = set(filtered_addresses)

with open('hacker_wallet.txt', 'w') as file:
    for sender in unique_senders:
        file.write(f"{sender}\n")

print(f"去重后的发送者数量: {len(unique_senders)}")

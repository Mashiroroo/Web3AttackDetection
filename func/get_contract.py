from web3 import Web3
from utils.get_fields import get_chain_and_tx
from utils.processor import Processor

processor = Processor(transaction=None, chain=None)
chain_list, tx_list = get_chain_and_tx()

contract_list = []
for chain, tx in zip(chain_list, tx_list):
    # print(chain, tx)
    processor.chain = chain
    processor.transaction = tx
    processor.rpc_node = processor.config['rpc_nodes'].get(chain, None)
    if processor.rpc_node:
        processor.w3 = Web3(Web3.HTTPProvider(processor.rpc_node))
    contract = processor.find_contract()
    contract_list.append(contract)
    print()

filtered_addresses = [address for address in contract_list if address is not None]

unique_contracts = set(filtered_addresses)

with open('contract.txt', 'w') as file:
    for contract in unique_contracts:
        file.write(f"{contract}\n")

print(f"去重后的合约数量: {len(unique_contracts)}")

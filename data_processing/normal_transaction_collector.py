import random
from tqdm import tqdm
from web3 import Web3
import csv
from utils.processor import Processor

processor = Processor(transaction=None, chain='ETH')

if not processor.w3.is_connected:
    print("Failed to connect to the Ethereum node.")
    exit()


# 函数判断是否为合约地址
def is_contract_address(address):
    try:
        code = processor.w3.eth.get_code(address)
        return len(code) > 2  # 合约地址的 code 长度通常大于 2（非合约地址则是0x）
    except Exception as e:
        print(f"Error fetching contract code for address {address}: {e}")
        return False


# 存储合约交易的样本列表和已遇到的合约地址
sampled_transactions = []
seen_contracts = set()  # 用于记录已收集的合约地址

latest_block = processor.w3.eth.block_number
start_block = latest_block
end_block = latest_block - 10000000
num = 100000

random_blocks = random.sample(range(end_block, start_block), start_block - end_block)

for block_num in tqdm(random_blocks, desc="Processing blocks", unit="block"):
    try:
        block = processor.w3.eth.get_block(block_num, full_transactions=True)
    except Exception as e:
        print(f"Error fetching block {block_num}: {e}")
        continue  # 如果遇到区块获取失败，则跳过该区块

    for tx in block.transactions:
        # 如果交易的 to 地址是一个合约地址，且该合约地址未被处理过
        if tx.to and is_contract_address(tx.to) and tx.to not in seen_contracts:
            # print(tx.to)
            sampled_transactions.append({
                'chain': 'ETH',
                'blockNumber': block_num,
                'transactionHash': '0x' + tx.hash.hex()
            })
            seen_contracts.add(tx.to)  # 将合约地址加入已处理集合
            break
    if len(sampled_transactions) >= num:
        break

with open('normal_transactions.csv', 'w', newline='') as csvfile:
    fieldnames = ['chain', 'blockNumber', 'transactionHash']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(sampled_transactions)

print(f"Collected {len(sampled_transactions)} unique normal contract-interacting transactions.")

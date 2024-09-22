import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from utils.processor import Processor

rpc_nodes = [
    'https://cloudflare-eth.com',
    'https://ethereum.blockpi.network/v1/rpc/4ca6dcb6a65b915676a8f0b7246a4839086c6dd7',
    'https://ethereum-mainnet.core.chainstack.com/8edf135d4f7a63e93b69c44decc7a538',
    'https://go.getblock.io/2f9bba8389884e98b3a23fa727d50980',
    'https://eth.api.onfinality.io/public',
    'https://1rpc.io/eth',
    'https://rpc.flashbots.net',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=Asv3nQwPQE6AqrmVKYZBEgAFdL22d-8R74-_hlDYfw4q',
    'https://rpc.ankr.com/eth'
]

random.seed(123)


def get_random_rpc_node():
    return random.choice(rpc_nodes)


def create_processor(retries=3):
    for _ in range(retries):
        rpc_url = get_random_rpc_node()
        processor = Processor(transaction=None, chain=None, rpc_node=rpc_url)
        if processor.w3.is_connected():
            return processor
        print(f"Failed to connect to the Ethereum node: {rpc_url}. Retrying...")
        time.sleep(1)
    raise Exception("Unable to connect to any Ethereum node.")


progress_file = 'progress.json'
address_cache = {}


def is_contract_address(processor, address):
    if address in address_cache:
        return address_cache[address]
    try:
        code = processor.w3.eth.get_code(address)
        is_contract = len(code) > 2
        address_cache[address] = is_contract
        time.sleep(0.1)
        return is_contract
    except Exception as e:
        print(f"Error fetching contract code for address {address}: {e}")
        address_cache[address] = False
        return False


def process_block(block_num):
    processor = create_processor()
    try:
        block = processor.w3.eth.get_block(block_num, full_transactions=True)
        for tx in block.transactions:
            if tx.to and is_contract_address(processor, tx.to):
                return {'blockNumber': block_num, 'transactionHash': '0x' + tx.hash.hex()}
    except Exception as e:
        print(f"Error fetching block {block_num}: {e}")
    return None


processor = create_processor()

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    last_processed_block = progress['last_processed_block']
    # 读取当前已处理的交易数量
    current_processed_count = progress.get('transaction_count', 0)
else:
    last_processed_block = processor.w3.eth.block_number
    current_processed_count = 0  # 初始为 0

latest_block = processor.w3.eth.block_number
end_block = latest_block - 1000000

if last_processed_block - end_block > 0:
    random_blocks = random.sample(range(end_block, last_processed_block), last_processed_block - end_block)
else:
    print("区块范围无效，请调整 end_block 的值。")
    exit()

target_transaction_count = 100000
sampled_transactions = []

with open('transactions.csv', 'a', newline='') as csvfile:
    fieldnames = ['blockNumber', 'transactionHash']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    if os.stat('transactions.csv').st_size == 0:
        writer.writeheader()

    # 设置 tqdm 的总量为目标交易数量，并更新已处理数量
    with tqdm(total=target_transaction_count, initial=current_processed_count, desc="Processing transactions") as pbar:
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 10) as executor:
            future_to_block = {executor.submit(process_block, block_num): block_num for block_num in random_blocks}

            for future in as_completed(future_to_block):
                block_num = future_to_block[future]
                try:
                    result = future.result()
                    if result:
                        sampled_transactions.append(result)
                        writer.writerow(result)

                        # 更新进度条
                        pbar.update(1)

                    if len(sampled_transactions) >= target_transaction_count:
                        print(f"Reached target of {target_transaction_count} transactions.")
                        break

                    # 每隔几块保存一次进度
                    if block_num % 100 == 0:
                        with open(progress_file, 'w') as f:
                            json.dump({
                                'last_processed_block': block_num,
                                'transaction_count': len(sampled_transactions)
                            }, f)
                except Exception as e:
                    print(f"处理区块 {block_num} 时出错: {e}")

# 最终保存进度
with open(progress_file, 'w') as f:
    json.dump({
        'last_processed_block': last_processed_block,
        'transaction_count': len(sampled_transactions)
    }, f)

print(f"Collected {len(sampled_transactions)} unique contract-interacting transactions.")

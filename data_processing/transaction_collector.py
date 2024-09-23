import argparse
import csv
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
    # 'https://rpc.flashbots.net',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=Asv3nQwPQE6AqrmVKYZBEgAFdL22d-8R74-_hlDYfw4q',
    'https://rpc.ankr.com/eth',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al63O0vVeKgR74_ghlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al5vgPg9eKgR74_chlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6WvxNheKgR74_dhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al7c4vJFeKgR74_khlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al7S1iypeKgR74_jhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6vcdfheKgR74_fhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al4IyDM-d_sR74_ChlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=Asv3nQwPQE6AqrmVKYZBEgAFdL22d-8R74-_hlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6_PDK6eKgR74_hhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6oHm4xeKgR74_ehlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al5eQjrPeK8R74_lhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al5eQjrPeK8R74_lhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6c1n1BeK8R74_nhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6mTzM_eK8R74_ohlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6s4AGeeK8R74_phlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6ygV_seK8R74_qhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al7J_mDHeK8R74_rhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al7W-snueK8R74_thlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al7g3OOreK8R74_uhlDYfw4q'
]

random.seed(123)
node_index = 0


def get_next_rpc_node():
    global node_index
    rpc_node = rpc_nodes[node_index]
    node_index = (node_index + 1) % len(rpc_nodes)
    return rpc_node


def create_processor(retries=3):
    for _ in range(retries):
        rpc_url = get_next_rpc_node()
        processor = Processor(transaction=None, chain=None, rpc_node=rpc_url)
        if processor.w3.is_connected():
            return processor
        print(f"Failed to connect to the Ethereum node: {rpc_url}. Retrying...")
        time.sleep(1)
    time.sleep(10)
    raise Exception("Unable to connect to any Ethereum node.")


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


def process_block_for_contracts(block_num):
    processor = create_processor()
    try:
        block = processor.w3.eth.get_block(block_num, full_transactions=True)
        for tx in block.transactions:
            if tx.to and is_contract_address(processor, tx.to):
                return {'blockNumber': block_num, 'transactionHash': '0x' + tx.hash.hex()}
    except Exception as e:
        print(f"Error fetching block {block_num}: {e}")
    return None


def process_block_for_random(block_num):
    processor = create_processor()
    try:
        block = processor.w3.eth.get_block(block_num, full_transactions=True)
        if block.transactions:
            tx = random.choice(block.transactions)
            return {'blockNumber': block_num, 'transactionHash': '0x' + tx.hash.hex()}
    except Exception as e:
        print(f"Error fetching block {block_num}: {e}")
    return None


def get_transaction_count(csv_file):
    if not os.path.exists(csv_file):
        return 0
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        row = sum(1 for _ in reader)
        return row - 1 if row > 0 else 0


def get_last_processed_block(csv_file):
    if not os.path.exists(csv_file):
        return None
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            last_row = row
        return int(last_row['blockNumber']) if last_row else None


def collect_transactions(mode, output, target_transaction_count):
    processor = create_processor()

    latest_block = processor.w3.eth.block_number
    end_block = 1000000

    last_processed_block = get_last_processed_block(output)
    if last_processed_block is None:
        last_processed_block = latest_block

    if last_processed_block - end_block > 0:
        random_blocks = random.sample(range(end_block, latest_block), target_transaction_count * 5)
    else:
        print("区块范围无效，请调整 end_block 的值。")
        exit()

    current_processed_count = get_transaction_count(output)

    with open(output, 'a', newline='') as csvfile:
        fieldnames = ['blockNumber', 'transactionHash']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if os.stat(output).st_size == 0:
            writer.writeheader()

        with tqdm(total=target_transaction_count, initial=current_processed_count,
                  desc="Processing transactions") as pbar:
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 10) as executor:
                if mode == 'contract':
                    future_to_block = {executor.submit(process_block_for_contracts, block_num): block_num for block_num
                                       in random_blocks}
                else:
                    future_to_block = {executor.submit(process_block_for_random, block_num): block_num for block_num in
                                       random_blocks}

                for future in as_completed(future_to_block):
                    block_num = future_to_block[future]
                    try:
                        result = future.result()
                        if result:
                            writer.writerow(result)
                            pbar.update(1)

                        if get_transaction_count(output) >= target_transaction_count:
                            print(f"Reached target of {target_transaction_count} transactions.")
                            # executor.shutdown(wait=False)
                            break

                    except Exception as e:
                        print(f"处理区块 {block_num} 时出错: {e}")

    print(f"Collected {get_transaction_count(output)} unique transactions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Ethereum transactions")
    parser.add_argument(
        '--mode', choices=['contract', 'random'], required=True,
        help="Mode: 'contract' to collect contract transactions, 'random' to collect random transactions"
    )
    parser.add_argument(
        '--output', type=str, default='transactions.csv',
        help="Output CSV file for collected transactions"
    )
    parser.add_argument(
        '--count', type=int, default=50000,
        help="Target number of transactions to collect"
    )

    args = parser.parse_args()

    collect_transactions(args.mode, args.output, args.count)

    '''
    python transaction_collector.py --mode contract --output contract_transactions.csv --count 100000
    python -m data_processing.transaction_collector --mode random --output random_transactions.csv --count 50000
    '''

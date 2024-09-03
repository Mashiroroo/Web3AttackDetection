import os
import time
import logging
from web3 import Web3
from utils.get_fields import get_chain_and_tx
from utils.processor import Processor

logging.basicConfig(level=logging.INFO)
chain_list, tx_list = get_chain_and_tx()
loader = Processor(transaction=None, chain=None)

log_file = 'processed_transactions.log'
error_log_file = 'failed_transactions.log'

if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        processed = set(line.strip() for line in f)
else:
    processed = set()

if os.path.exists(error_log_file):
    with open(error_log_file, 'r') as f:
        failed = set(line.strip() for line in f)
else:
    failed = set()


def log_processed_transaction(chain, tx):
    with open(log_file, 'a') as f:
        f.write(f'{chain}:{tx}\n')


def log_failed_transaction(chain, tx):
    with open(error_log_file, 'a') as f:
        f.write(f'{chain}:{tx}\n')


def load_transaction_with_retry(chain, tx, retries=2, delay=10):
    loader.chain = chain
    loader.transaction = tx
    loader.rpc_node = loader.config['rpc_nodes'].get(chain, None)
    if loader.rpc_node:
        loader.w3 = Web3(Web3.HTTPProvider(loader.rpc_node))
    else:
        print(f'No RPC node for chain {chain}')

    for attempt in range(retries):
        try:
            loader.load_transaction_trace(output_dir='../tx_trace')
            logging.info(f"Successfully loaded transaction {tx} on chain {chain}")
            log_processed_transaction(chain, tx)  # 记录已处理交易
            return True
        except Exception as e:
            logging.error(
                f"Error loading transaction {tx} on chain {chain}: {e}. Retry {attempt + 1}/{retries}")
            time.sleep(delay)
    logging.error(f"Failed to load transaction {tx} on chain {chain} after {retries} attempts")
    # log_processed_transaction(chain, tx)
    log_failed_transaction(chain, tx)  # 记录失败交易
    return False


for chain, tx in zip(chain_list, tx_list):
    if f'{chain}:{tx}' in processed:
        print(f'{chain}:{tx} is already processed.')
        continue
    elif f'{chain}:{tx}' in failed:
        print(f'{chain}:{tx} is already failed.')
        continue

    success = load_transaction_with_retry(chain, tx)
    if not success:
        continue
    time.sleep(5)  # 根据节点情况增加时间

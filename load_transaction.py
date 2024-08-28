import time
import logging
from web3 import Web3
from utils.get_fields import get_chain_and_tx
from utils.processor import Processor
from utils.get_tx_hash import get_tx_hash

# 配置日志记录
logging.basicConfig(level=logging.INFO)
chain_list, tx_list = get_chain_and_tx(r'./dataset/utf8Format_standard.csv')
loader = Processor(None, None, None, config_path=r'config.yaml')


def load_transaction_with_retry(loader, chain, tx, retries=3, delay=5):
    loader.chain = chain
    loader.transaction = get_tx_hash(tx)
    loader.rpc_node = loader.config['rpc_nodes'].get(chain, None)

    if loader.rpc_node:
        loader.w3 = Web3(Web3.HTTPProvider(loader.rpc_node))

    for attempt in range(retries):
        try:
            loader.load_transaction(output_dir='data')
            logging.info(f"Successfully loaded transaction {tx} on chain {chain}")
            return True
        # except TransactionNotFound as e:
        #     logging.warning(f"Transaction not found: {e}. Retry {attempt + 1}/{retries}")
        except Exception as e:
            print(e)
            logging.error(f"Error loading transaction {tx} on chain {chain}: {e}. Retry {attempt + 1}/{retries}")
        time.sleep(delay)
    logging.error(f"Failed to load transaction {tx} on chain {chain} after {retries} attempts")
    return False


for chain, tx in zip(chain_list, tx_list):
    success = load_transaction_with_retry(loader, chain, tx)
    if not success:
        continue  # 记录失败的交易并跳过
    time.sleep(5)  # 动态调整时间或根据实际情况增加时间

# # 设置最大线程数
# max_workers = 1
#
# with ThreadPoolExecutor(max_workers=max_workers) as executor:
#     # 提交所有任务到线程池
#     futures = [executor.submit(process_transaction, chain, tx) for chain, tx in zip(chain_list, tx_list)]
#
#     # 等待所有线程完成并获取结果
#     for future in concurrent.futures.as_completed(futures):
#         try:
#             future.result()  # 可以在这里处理每个线程的结果或异常
#         except Exception as e:
#             print(f"Error processing transaction: {e}")

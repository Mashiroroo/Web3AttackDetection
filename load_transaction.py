import csv
import time

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
    time.sleep(5)

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

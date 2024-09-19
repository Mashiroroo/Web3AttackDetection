import json
import os
import re

from tqdm import tqdm
from web3 import Web3
from trace.trace_debug import parse_trace
from utils.processor import Processor
from utils.get_fields import get_chain_by_tx

trace_data_dir = r'../trace_data'
processor = Processor(transaction=None, chain=None)


def get_all_addr(data_dir):
    result = []

    # 遍历trace_data
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)

            with open(file_path, 'r', encoding='utf8') as f:
                trace_data = f.readlines()

            address_pattern = re.compile(r'0x[a-fA-F0-9]{40}')
            involved_addresses = set()

            for line in trace_data:
                addresses = address_pattern.findall(line)
                if addresses:
                    for address in addresses:
                        try:
                            # 将普通地址转换为校验和地址
                            checksum_address = Web3.to_checksum_address(address)
                            involved_addresses.add(checksum_address)
                        except ValueError:
                            # 如果地址无效，跳过该地址
                            print(f"Invalid address found: {address}")
                            continue

            # 从文件路径中提取交易哈希
            tx_hash = file_path[-70:-4]

            # 将交易哈希和相关地址存储到结果中
            result.append({
                'tx_hash': tx_hash,
                'addresses': list(involved_addresses)
            })

    return result  # [{'tx_hash':..., addresses:[.......]}]


def get_balance_changes(data_dir):
    all_addr_dict_list = get_all_addr(trace_data_dir)
    # result = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            tx_hash = file_path[-70:-4]
            processor.transaction = tx_hash
            processor.chain = get_chain_by_tx(tx_hash)
            processor.rpc_node = processor.config['rpc_nodes'].get(processor.chain, None)
            if processor.rpc_node:
                processor.w3 = Web3(Web3.HTTPProvider(processor.rpc_node))
            for item in tqdm(all_addr_dict_list):  # 一个交易的，{tx_hash:..., addresses:[...]}
                # print(item)
                res = {
                    "tx_hash": tx_hash,
                    "addresses": list(processor.get_balance(item)),
                }
                with open(f'{tx_hash}.json', 'w', encoding='utf8') as f:
                    json.dump(res, f, ensure_ascii=False, indent=4)
                # result.append(res)
                # print(result)
    # return result


if __name__ == '__main__':
    res = get_balance_changes(trace_data_dir)
    print(res)

    # print(address_list)
    # balance_changes = parse_trace(file_path)
    # # print(balance_changes)
    # address_increase = []
    # for addr, change in balance_changes.items():
    #     # print(f"Address: {addr}, Balance Change: {change}")
    #     if change > 0:
    #         address_increase.append({'addr': addr, 'change': change})
    # chain = get_chain_by_tx(tx_hash)
    # # print(f"链：{chain}")
    # processor.chain = chain
    # processor.transaction = tx_hash
    # processor.rpc_node = processor.config['rpc_nodes'].get(chain, None)
    # if processor.rpc_node:
    #     processor.w3 = Web3(Web3.HTTPProvider(processor.rpc_node))
    # else:
    #     print(f'No RPC node for chain {chain}')
    # res = processor.get_ex_balance(address_increase)
    # print(res)
    # output_file = r'sus_wallet.txt'
    # with open(output_file, 'a') as f:
    #     if res is not None:
    #         for item in res:
    #             f.write(item['addr'] + '\n')
    # print('------------------------------------------------------------------------------------------------------')

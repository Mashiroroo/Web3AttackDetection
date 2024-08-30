import os

import requests
from web3 import Web3
import yaml
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))


class Processor:
    def __init__(self, rpc_node, transaction, chain):
        self.config_path = os.path.join(current_dir, '../config.yaml')
        self.config = self.load_config()
        self.transaction = transaction
        self.chain = chain
        self.rpc_node = rpc_node or self.config['rpc_nodes'].get(chain, None)
        if self.rpc_node is not None:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_node))
        else:
            self.w3 = None

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            print(f"loading config: {config}")
        return config

    def find_sender(self):
        if self.w3 is not None:
            try:
                tx = self.w3.eth.get_transaction(self.transaction)
                from_address = tx['from']
                print(f'在{self.chain}上的交易 {self.transaction} from: {from_address}')
                return from_address
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No Web3 instance available.")

    def load_transaction_trace(self, output_dir):
        output_file = f"{output_dir}/transaction_{self.transaction}.txt"
        cast_command = f"{self.config['cast_path']} run {self.transaction} --rpc-url {self.rpc_node}"
        timeout_seconds = 30
        try:
            with open(output_file, 'w') as f:
                result = subprocess.run(
                    [self.config['git_bash_path'], '-c', cast_command],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_seconds
                )
            # print(f"Return Code: {result.returncode}")
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
        except subprocess.TimeoutExpired:
            print(f"Command timed out after {timeout_seconds} seconds.")

    # 输入balance增加的字典（地址：增加的钱），返回block-1上balance为0的地址的字典
    def get_ex_balance(self, addresses_dict_list):
        try:
            tx_receipt = self.w3.eth.get_transaction_receipt(self.transaction)
            block_number = tx_receipt['blockNumber']
            # 计算 block_number - 1
            previous_block = block_number - 1

            balances = {}
            # addresses_dict_list是 一个交易 在block上的[{adr1:balance1},{adr2:balance2},...]
            if addresses_dict_list is not None:
                for item in addresses_dict_list:
                    try:
                        balance = self.w3.eth.get_balance(item['addr'], block_identifier=previous_block)
                        balances[item['addr']] = self.w3.from_wei(balance, 'wei')
                    except Exception as e:
                        print(e)
                print(f"Block-1: {previous_block}")
                # balances是 这个交易 在block - 1 上的[{adr1:balance1},{adr2:balance2},...]
                sus_list = []
                for addr, balance in balances.items():
                    print(f"Address: {addr}, Previous Balance: {balance}")
                    if balance == 0:
                        sus_list.append(addr)
                    # 过滤出 addresses_dict_list 中地址属于 sus_list 的项
                    filtered_addresses = [item for item in addresses_dict_list if item['addr'] in sus_list]
                    # 按 block 上的 change 从大到小排序
                    sorted_filtered_addresses = sorted(filtered_addresses, key=lambda x: x['change'], reverse=True)
                    # 返回 之前金额是几乎是0，同时金额增加的 排序下 top2
                    return sorted_filtered_addresses[:2]

        except Exception as e:
            print(e)


if __name__ == '__main__':
    processor = Processor(rpc_node=None,
                          transaction='0x93ae5f0a121d5e1aadae052c36bc5ecf2d406d35222f4c6a5d63fef1d6de1081',
                          chain='BSC')
    # processor.find_sender()

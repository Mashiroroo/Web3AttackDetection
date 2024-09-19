import logging
import os
from web3 import Web3
import yaml
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))


class Processor:
    def __init__(self, transaction, chain):
        self.config_path = os.path.join(current_dir, '../config.yaml')
        self.config = self.load_config()
        self.transaction = transaction
        self.chain = chain
        self.rpc_node = self.config['rpc_nodes'].get(chain, None)
        if self.rpc_node is not None:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_node))
        else:
            self.w3 = None

        self.balance_dict_list = []
        self.ex_balance_dict_list = []

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            print(f"loading config: {config}")
        return config

    # 使用下面方法，不是基于eth的链将无法加载
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

    # 使用下面方法，不是基于eth的链将无法加载
    def find_contract(self):
        if self.w3 is not None:
            try:
                tx = self.w3.eth.get_transaction(self.transaction)
                contract_address = tx['to']
                print(f'在{self.chain}上的交易 {self.transaction} to: {contract_address}')
                if contract_address is None:
                    if len(tx['input']) < 10:
                        # print(tx['input'])
                        return None
                    try:
                        receipt = self.w3.eth.get_transaction_receipt(self.transaction)
                        if receipt['status'] == 1:
                            # print(receipt)
                            if 'contractAddress' in receipt and receipt['contractAddress']:
                                print(
                                    f"发现创建合约地址: {receipt['contractAddress']}, 区块编号: {receipt['blockNumber']}")

                    except Exception as err:
                        logging.error(f"获取交易 receipt 出错，使用区块 receipts: {err}, {self.transaction}")
                        # receipts = receipt['log']
                return contract_address
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No Web3 instance available.")

    def load_transaction_trace(self, output_dir):
        output_file = f"{output_dir}/transaction_{self.transaction}.txt"
        cast_command = f"{self.config['cast_path']} run {self.transaction} --rpc-url {self.rpc_node}"
        timeout_seconds = 600
        with open(output_file, 'w') as f:
            result = subprocess.run(
                [self.config['git_bash_path'], '-c', cast_command],
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                timeout=timeout_seconds
            )
        # print(f"Return Code: {result.returncode}")
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    # 输入balance增加的字典（地址：增加的钱），返回block-1上balance为0的地址的字典

    def get_balance(self, address_dict):
        try:
            tx_receipt = self.w3.eth.get_transaction_receipt(self.transaction)
            block_number = tx_receipt['blockNumber']
            if address_dict is not None:
                for address in address_dict['addresses']:
                    try:
                        balance = self.w3.eth.get_balance(address, block_identifier=block_number)
                        # print(balance)
                        res_dict = {address: balance}
                        self.balance_dict_list.append(res_dict)
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)
        # print(self.balance_dict_list)
        return self.balance_dict_list

    def get_ex_balance(self, addresses_dict):
        try:
            tx_receipt = self.w3.eth.get_transaction_receipt(self.transaction)
            block_number = tx_receipt['blockNumber']
            # 计算 block_number - 1
            previous_block = block_number - 1
            if addresses_dict is not None:
                for address in addresses_dict['addresses']:
                    try:
                        balance = self.w3.eth.get_balance(address, block_identifier=previous_block)
                        res_dict = {address: balance}
                        self.ex_balance_dict_list.append(res_dict)
                    except Exception as e:
                        print(e)
                print(f"Block-1: {previous_block}")
        except Exception as e:
            print(e)
        return self.ex_balance_dict_list


if __name__ == '__main__':
    processor = Processor(
        transaction='0x93ae5f0a121d5e1aadae052c36bc5ecf2d406d35222f4c6a5d63fef1d6de1081',
        chain='BSC')
    # processor.find_sender()

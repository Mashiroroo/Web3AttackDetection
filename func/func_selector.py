from web3 import Web3

from utils.get_fields import get_chain_and_tx
from utils.processor import Processor

processor = Processor(transaction=None, chain=None)


def main():
    chain_list, tx_list = get_chain_and_tx()
    for chain, tx in zip(chain_list, tx_list):
        processor.chain = chain
        processor.transaction = tx
        processor.rpc_node = processor.config['rpc_nodes'].get(chain, None)
        if processor.rpc_node:
            processor.w3 = Web3(Web3.HTTPProvider(processor.rpc_node))
        contract_address = get_contract_address(chain, tx)
        if contract_address is not None:
            # 获取字节码
            bytecode = get_contract_bytecode(contract_address)
            # 提取选择器
            selectors = get_selectors_from_bytecode(bytecode)
            # 打印选择器
            for selector in selectors:
                print(selector)
            print('---------------------------------------------------------------------------------------------------')


def get_contract_address(chain, tx):
    contract_address = processor.find_contract()
    return contract_address


def get_contract_bytecode(contract_address):
    # 获取合约的字节码
    bytecode = processor.w3.eth.get_code(contract_address)
    return bytecode


def get_selectors_from_bytecode(bytecode):
    selectors = set()  # 使用集合来避免重复的选择器
    bytecode_length = len(bytecode)

    # 遍历字节码，查找 4 字节的选择器
    i = 0
    while i < bytecode_length - 4:
        if bytecode[i] == 0x60:  # PUSH1 指令的前缀
            i += 1
            continue
        elif bytecode[i] == 0x7f:  # PUSH4 指令的前缀
            selector = bytecode[i + 1:i + 5]  # 获取接下来的 4 字节
            selectors.add(selector.hex())
            i += 5
        else:
            i += 1

    return selectors


if __name__ == "__main__":
    main()

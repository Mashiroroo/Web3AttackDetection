import re
from utils.get_fields import get_chain_and_tx


def get_address(tx_list):
    for tx in tx_list:
        file_path = f'../trace_data/transaction_{tx}.txt'
        try:
            with open(file_path, 'r', encoding='utf8') as file:
                trace_data = file.readlines()
        except FileNotFoundError:
            continue
        address_pattern = re.compile(r'0x[a-fA-F0-9]{40}')
        involved_addresses = set()
        for line in trace_data:
            addresses = address_pattern.findall(line)
            if addresses:
                for address in addresses:
                    involved_addresses.add(address)
        print(f"交易{tx}涉及到的地址：")
        for address in involved_addresses:
            print(address)
        print('-------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    chain_list, tx_list = get_chain_and_tx()
    get_address(tx_list)

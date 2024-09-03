import re
from collections import defaultdict

# 用于存储每个地址的余额变化
balance_changes = defaultdict(int)


def parse_trace(trace_data_file):
    with open(trace_data_file, 'r', encoding='utf-8') as file:
        trace_data = file.read()
    balance_changes = defaultdict(int)

    transfer_pattern = re.compile(
        r'emit Transfer\(param0: (0x[0-9a-fA-F]{40}), param1: (0x[0-9a-fA-F]{40}), param2: (\d+)')
    balance_of_pattern = re.compile(r'balanceOf\((0x[0-9a-fA-F]{40})\)')
    eth_transfer_pattern = re.compile(
        r'call: to=(0x[0-9a-fA-F]{40}), value=(\d+)')

    for line in trace_data.splitlines():
        # 查找Transfer事件
        transfer_match = transfer_pattern.search(line)
        if transfer_match:
            from_addr, to_addr, amount = transfer_match.groups()
            amount = int(amount)

            # 更新发送方和接收方的余额变化
            balance_changes[from_addr] -= amount
            balance_changes[to_addr] += amount

        # 查找balanceOf调用
        balance_of_match = balance_of_pattern.search(line)
        if balance_of_match:
            addr = balance_of_match.group(1)
            # 这里你可以进一步处理balanceOf相关的数据，如需要

        # 查找ETH转账操作
        eth_transfer_match = eth_transfer_pattern.search(line)
        if eth_transfer_match:
            to_addr, value = eth_transfer_match.groups()
            value = int(value)

            # ETH转账只影响接收方的余额
            balance_changes[to_addr] += value

    return balance_changes


if __name__ == '__main__':
    # 示例文件路径
    file_path = r'../tx_trace/transaction_0x906394b2ee093720955a7d55bff1666f6cf6239e46bea8af99d6352b9687baa4.txt'

    balance_changes = parse_trace(file_path)

    for addr, change in balance_changes.items():
        print(f"Address: {addr}, Balance Change: {change}")

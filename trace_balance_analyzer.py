import re
from collections import defaultdict

# 用于存储每个地址的余额变化
balance_changes = defaultdict(int)


def parse_trace(trace_data):
    # 匹配Transfer事件或balanceOf调用的正则表达式
    transfer_pattern = re.compile(
        r'emit Transfer\(param0: (0x[0-9a-fA-F]{40}), param1: (0x[0-9a-fA-F]{40}), param2: (\d+)')
    balance_of_pattern = re.compile(r'balanceOf\((0x[0-9a-fA-F]{40})\)')

    # 遍历trace中的每一行
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

    return balance_changes


# 从txt文件中读取trace字符串
def read_trace_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        trace = file.read()
    return trace


# 示例文件路径
file_path = r'./data/transaction_0xe1f257041872c075cbe6a1212827bc346df3def6d01a07914e4006ec43027165.txt'

# 从文件中读取交易追踪信息
trace = read_trace_from_file(file_path)

# 解析交易追踪信息
balance_changes = parse_trace(trace)

# 打印每个地址的余额变化
for addr, change in balance_changes.items():
    print(f"Address: {addr}, Balance Change: {change}")

from web3 import Web3

# 初始化Web3
w3 = Web3(Web3.HTTPProvider('https://docs-demo.optimism.quiknode.pro/'))

# 交易hash
tx_hash = "0xb43748ed668c1e44cf0a3e829ca0fe24eceaee7d33d06072bb11ca99afa7f448"

# 获取交易的事件日志
receipt = w3.eth.get_transaction_receipt(tx_hash)

# ERC20 Transfer 事件的签名哈希
TRANSFER_EVENT_SIGNATURE_HASH = w3.keccak(text="Transfer(address,address,uint256)").hex()

# 存储地址与资金变化的字典
balance_changes = {}

# 解析日志中的 Transfer 事件
for log in receipt["logs"]:
    print(log)
    # 如果是 ERC20 的 Transfer 事件
    if log["topics"][0].hex() == TRANSFER_EVENT_SIGNATURE_HASH:
        from_addr = Web3.to_checksum_address("0x" + log["topics"][1].hex()[-40:])  # 发送地址
        to_addr = Web3.to_checksum_address("0x" + log["topics"][2].hex()[-40:])  # 接收地址

        # 将字节串转换为整数
        value = int.from_bytes(log["data"], byteorder='big')  # 转账的代币数量

        # 更新发送方资金变化
        if from_addr not in balance_changes:
            balance_changes[from_addr] = 0
        balance_changes[from_addr] -= value

        # 更新接收方资金变化
        if to_addr not in balance_changes:
            balance_changes[to_addr] = 0
        balance_changes[to_addr] += value

# 输出结果
print("涉及的地址及资金变化（ERC20 代币）：")
for address, change in balance_changes.items():
    change_tokens = change / 1e18  # 假设代币有 18 位小数
    print(f"地址: {address}, 资金变化: {change_tokens:.6f} 代币")

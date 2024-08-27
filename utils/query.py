from web3 import Web3


class Query:
    def __init__(self, rpc_node, transaction, chain):
        self.rpc_node_dist = {
            'ETH': 'https://rpc.ankr.com/eth',
            'BNB': 'https://rpc.ankr.com/bsc',
            'BSC': 'https://rpc.ankr.com/bsc',
            'Polygon': 'https://rpc.ankr.com/polygon',
            'Arbitrum': 'https://rpc.ankr.com/arbitrum',
            'Optimism': 'https://rpc.ankr.com/optimism',
            'Avalanch': 'https://rpc.ankr.com/avalanche',
            'Base': 'https://rpc.ankr.com/base',
            'Mantle': 'https://rpc.ankr.com/mantle',
            'Avalanche': 'https://rpc.ankr.com/avalanche',
            'TRON': 'https://rpc.ankr.com/http/tron'
        }
        self.transaction = transaction
        self.chain = chain
        self.rpc_node = rpc_node or self.rpc_node_dist.get(chain, None)
        if self.rpc_node:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_node))
        else:
            print(f"No valid RPC node found for chain: {chain}")
            self.w3 = None

    def find_sender(self):
        if self.w3 is not None:
            try:
                tx = self.w3.eth.get_transaction(self.transaction)
                from_address = tx['from']
                print(f'交易 {self.transaction} sender 的地址是: {from_address}')
                return from_address
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No Web3 instance available.")


if __name__ == '__main__':
    query = Query(rpc_node=None, transaction='0x93ae5f0a121d5e1aadae052c36bc5ecf2d406d35222f4c6a5d63fef1d6de1081',
                  chain='BSC')
    query.find_sender()

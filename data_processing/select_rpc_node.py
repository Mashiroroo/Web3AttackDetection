from utils.processor import Processor

rpc_nodes = [
    'https://rpc.ankr.com/eth',
    'https://ethereum.blockpi.network/v1/rpc/4ca6dcb6a65b915676a8f0b7246a4839086c6dd7',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=Asv3nQwPQE6AqrmVKYZBEgAFdL22d-8R74-_hlDYfw4q',
    'https://cloudflare-eth.com',
    'https://api.blocknative.com/v0/jsonrpc',
    'https://nd-123-456-789.p2pify.com/',
    'https://eth-mainnet.blastapi.io/',
    'https://nodes.mewapi.io/rpc/eth',
    'https://rpc.flashbots.net',
    'https://eth.mainnet.public.allnodes.com',
    'https://1rpc.io/eth',
    'https://eth.api.onfinality.io/public',
    'https://rpc.gw.fm',
    'https://api.covalenthq.com/v1/pricing/ETH/',
    'https://ethereum.public-rpc.gsr.io',
    'https://ethereum-mainnet.core.chainstack.com/8edf135d4f7a63e93b69c44decc7a538',
    'https://ethereum-mainnet.core.chainstack.com/beacon',
    'https://ethereum-mainnet.core.chainstack.com',
    'https://go.getblock.io/2f9bba8389884e98b3a23fa727d50980'
]

rpc_res_node = []


def create_processor(retry=5):
    for i in range(retry):
        try:
            for node in rpc_nodes:
                processor = Processor(chain=None, transaction=None, rpc_node=node)
                if processor.w3.is_connected():
                    rpc_res_node.append(node)
        except Exception as e:
            print(node)
    return rpc_res_node


if __name__ == '__main__':
    res = create_processor()
    print(list(set(res)))

from web3 import Web3
import yaml
import subprocess


class Processor:
    def __init__(self, rpc_node, transaction, chain, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        self.transaction = transaction
        self.chain = chain
        self.rpc_node = rpc_node or self.config['rpc_nodes'].get(chain, None)
        if self.rpc_node:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_node))
        else:
            print(f"No valid RPC node found for chain: {chain}")
            self.w3 = None

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            print(config)
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

    def load_transaction(self, output_dir):
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
            print(f"Return Code: {result.returncode}")
            if result.returncode != 0:
                print(f"Error Output: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Command timed out after {timeout_seconds} seconds.")
        except Exception as e:
            print(f"Exception occurred: {e}")


if __name__ == '__main__':
    query = Processor(rpc_node=None, transaction='0x93ae5f0a121d5e1aadae052c36bc5ecf2d406d35222f4c6a5d63fef1d6de1081',
                      chain='BSC', config_path=r'../config.yaml')
    # query.find_sender()

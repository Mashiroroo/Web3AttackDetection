import csv
import json
import time

import requests
from fake_useragent import UserAgent
from tqdm import tqdm

ua = UserAgent()


# 重试机制函数
def get_with_retries(url, headers, data, retries=3, delay=5, method='post'):
    for i in range(retries):
        try:
            if method == 'post':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # 检查请求是否成功
            return response.text
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Request failed: {e}. Retrying {i + 1}/{retries}...")
            time.sleep(delay)  # 等待一段时间后重试
        except requests.exceptions.RequestException as e:
            print(f"{tx_hash} Request failed: {e}")
            break
    return None


def get_chain_id(tx):
    url = f'https://app.blocksec.com/api/v1/website-search?query={tx}'
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data=None, retries=3, delay=5, method='get')
    if response_text is None:
        return None

    try:
        data_list = json.loads(response_text)
        if not data_list:
            print(f"No chainId found for transaction {tx}")
            return None
        return data_list[0]['chainId']
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Unexpected response format for tx {tx}: {e}")
        return None


def get_balance_change(tx, chain_id):
    url = "https://app.blocksec.com/api/v1/onchain/tx/balance-change"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    return response_text or "{}"


def get_profile(tx, chain_id):
    url = "https://app.blocksec.com/api/v1/onchain/tx/profile"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    if response_text is None:
        return None, None, None

    try:
        profile = json.loads(response_text)
        return profile['basicInfo'], profile['fundFlow'], profile['tokenInfos']
    except KeyError as e:
        print(f"Missing expected data in profile for tx {tx}: {e}")
        return None, None, None


def get_address_label(tx, chain_id):
    url = "https://app.blocksec.com/api/v1/onchain/tx/address-label"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    return response_text or "{}"


def get_trace(tx, chain_id):
    url = "https://app.blocksec.com/api/v1/onchain/tx/trace"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    return response_text or "{}"


def get_state_change(tx, chain_id):
    url = "https://app.blocksec.com/api/v1/onchain/tx/state-change"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    return response_text or "{}"


def get_combined_info(tx, chain_id):
    basic_info, fund_flow, token_infos = get_profile(tx, chain_id)
    return {
        "balance_change": json.loads(get_balance_change(tx, chain_id)),
        "profile": {
            "basic_info": basic_info,
            "fund_flow": fund_flow,
            "token_infos": token_infos,
        },
        "address_label": json.loads(get_address_label(tx, chain_id)),
        "trace": json.loads(get_trace(tx, chain_id)),
        "state_change": json.loads(get_state_change(tx, chain_id))
    }


def read_transaction_hashes(file_path):
    hashes = []
    with open(file_path, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if 'transactionHash' in row:
                hashes.append(row['transactionHash'])
    return hashes


def read_last_processed_hash(checkpoint_file):
    try:
        with open(checkpoint_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def write_last_processed_hash(checkpoint_file, tx_hash):
    with open(checkpoint_file, 'w') as f:
        f.write(tx_hash)


if __name__ == "__main__":
    chain_id = 1
    transaction_hashes = read_transaction_hashes('../random_transactions.csv')
    checkpoint_file = 'last_processed_hash.txt'
    last_processed_hash = read_last_processed_hash(checkpoint_file)
    start_index = transaction_hashes.index(last_processed_hash) + 1 if last_processed_hash in transaction_hashes else 0
    with tqdm(total=len(transaction_hashes), initial=start_index, desc="Processing Transactions") as pbar:
        for tx_hash in transaction_hashes[start_index:]:
            try:
                chain_id = 1
                combined_data = get_combined_info(tx_hash, chain_id)
                with open(f'normal_data/{tx_hash}.json', 'w') as json_file:
                    json.dump(combined_data, json_file, indent=4)
                write_last_processed_hash(checkpoint_file, tx_hash)
                pbar.update(1)
            except Exception as e:
                print(f"Error processing {tx_hash}: {e}")
                break

    # chain, tx = get_chain_and_tx()
    # for tx_hash in tqdm(tx):
    #     chain_id = get_chain_id(tx_hash)
    #     if chain_id is None:
    #         continue
    #
    #     combined_data = get_combined_info(tx_hash)
    #     with open(f'data/{tx_hash}.json', 'w') as json_file:
    #         json.dump(combined_data, json_file, indent=4)

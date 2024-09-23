import json
from tqdm import tqdm
from utils.get_fields import get_chain_and_tx
import requests
from fake_useragent import UserAgent
import time


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
    ua = UserAgent()
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


def get_balance_change(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/balance-change"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    if response_text is None:
        return "{}"  # 返回空数据以继续流程
    return response_text


def get_profile(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/profile"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    if response_text is None:
        return None, None, None  # 返回空数据以避免程序崩溃

    try:
        profile = json.loads(response_text)
        return profile['basicInfo'], profile['fundFlow'], profile['tokenInfos']
    except KeyError as e:
        print(f"Missing expected data in profile for tx {tx}: {e}")
        return None, None, None


def get_address_label(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/address-label"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    if response_text is None:
        return "{}"
    return response_text


def get_trace(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/trace"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    if response_text is None:
        return "{}"
    return response_text


def get_state_change(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/state-change"
    data = {"chainID": chain_id, "txnHash": tx, "blocked": False}
    head = {"User-Agent": ua.random}

    response_text = get_with_retries(url, head, data, retries=3, delay=5)
    if response_text is None:
        return "{}"
    return response_text


def get_combined_info(tx):
    return {
        "balance_change": json.loads(get_balance_change(tx)),
        "profile": {
            "basic_info": get_profile(tx)[0],
            "fund_flow": get_profile(tx)[1],
            "token_infos": get_profile(tx)[2],
        },
        "address_label": json.loads(get_address_label(tx)),
        "trace": json.loads(get_trace(tx)),
        "state_change": json.loads(get_state_change(tx))
    }


if __name__ == "__main__":
    chain_id = get_chain_id('0x5a86e1e738683a3e5d095fa34ae7592f6a08d172cc1fcb41c36751fe38c5b1a5')
    res = get_combined_info('0x5a86e1e738683a3e5d095fa34ae7592f6a08d172cc1fcb41c36751fe38c5b1a5')
    print(res)
    # chain, tx = get_chain_and_tx()
    # for tx_hash in tqdm(tx):
    #     chain_id = get_chain_id(tx_hash)
    #     if chain_id is None:
    #         continue
    #
    #     combined_data = get_combined_info(tx_hash)
    #     with open(f'data/{tx_hash}.json', 'w') as json_file:
    #         json.dump(combined_data, json_file, indent=4)

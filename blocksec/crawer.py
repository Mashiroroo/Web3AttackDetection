import json

from tqdm import tqdm

from utils.get_fields import get_chain_and_tx
import requests
from fake_useragent import UserAgent


def get_chain_id(tx):
    ua = UserAgent()
    url = f'https://app.blocksec.com/api/v1/website-search?query={tx}'
    head = {
        "User-Agent": ua.random
    }
    json_data = requests.get(url, headers=head).text
    data_list = json.loads(json_data)
    chain_id = data_list[0]['chainId']
    return chain_id


def get_balance_change(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/balance-change"
    data = {"chainID": chain_id, "txnHash": tx,
            "blocked": False}
    head = {
        "User-Agent": ua.random
    }
    balance_change = requests.post(url, json=data, headers=head).text
    print(balance_change)
    return balance_change


def get_profile(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/profile"
    data = {"chainID": chain_id, "txnHash": tx,
            "blocked": False}
    head = {
        "User-Agent": ua.random
    }
    profile = requests.post(url, json=data, headers=head).text
    profile = json.loads(profile)
    basic_info = profile['basicInfo']
    fund_flow = profile['fundFlow']
    security_event = profile['securityEvent']
    token_infos = profile['tokenInfos']
    return basic_info, fund_flow, security_event, token_infos


def get_address_label(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/address-label"
    data = {"chainID": chain_id, "txnHash": tx,
            "blocked": False}
    head = {
        "User-Agent": ua.random
    }
    address_lable = requests.post(url, json=data, headers=head).text
    return address_lable


def get_trace(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/trace"
    data = {"chainID": chain_id, "txnHash": tx,
            "blocked": False}
    head = {
        "User-Agent": ua.random
    }
    trace = requests.post(url, json=data, headers=head).text
    return trace


def get_state_change(tx):
    ua = UserAgent()
    url = "https://app.blocksec.com/api/v1/onchain/tx/state-change"
    data = {"chainID": chain_id, "txnHash": tx,
            "blocked": False}
    head = {
        "User-Agent": ua.random
    }
    state = requests.post(url, json=data, headers=head).text
    return state


def get_combined_info(tx):
    combined_info = {
        "balance_change": json.loads(get_balance_change(tx)),
        "profile": {
            "basic_info": get_profile(tx)[0],
            "fund_flow": get_profile(tx)[1],
            "security_event": get_profile(tx)[2],
            "token_infos": get_profile(tx)[3]
        },
        "address_label": json.loads(get_address_label(tx)),
        "trace": json.loads(get_trace(tx)),
        "state_change": json.loads(get_state_change(tx))
    }
    return combined_info


if __name__ == "__main__":
    chain, tx = get_chain_and_tx()
    for tx_hash in tqdm(tx):
        chain_id = get_chain_id(tx_hash)
        combined_data = get_combined_info(tx_hash)
        json.dumps(combined_data, indent=4)
        with open(f'data/{tx_hash}.json', 'w') as json_file:
            json.dump(combined_data, json_file, indent=4)

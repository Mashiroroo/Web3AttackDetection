import json
import os
import sys

from volcenginesdkarkruntime import Ark


def parse_tweets(tweet):
    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    completion = client.chat.completions.create(
        model="ep-20240903150101-vqg4p",
        messages=[
            {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
            {"role": "user", "content": f"""
            以下 是一个Web3安全相关的twitter内容，请分析判断是否是一个攻击(attack)或欺诈/钓鱼(scam/phishing)事件预警(alert)。如果是的话，请帮我从内容中提取以下字段，如果提取不到请留空：
            1. ti_type: 预警事件类型，attack 或 scam
            2. chain_id: 事件发生的链ID
            3. attacker: 事件的攻击者地址
            4. victim：事件的受害者地址
            5. tx_hash: 事件的所有相关交易hash
            6. loss_usd_amount: 事件造成的资金损失(以USD计)
            7. url ： 相关报道地址。
            要求以json 格式返回。
            其中，如果不是，ti_type 填充为0
            {tweet}
            """},
        ],
    )
    result = completion.choices[0].message.content
    print(result)
    output_file = 'parsed_tweets.json'

    if os.path.exists(output_file):
        # 如果文件存在，加载现有数据
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []

    data.append(json.loads(result))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parse_tweets(
        "{'created_at': 'Mon Sep 02 22:48:29 +0000 2024', 'favorite_count': 2, 'full_text': '🚨 In August 2024, Web3 security incidents caused losses exceeding $316M, with significant impacts on platforms like Ronin, Nexera, Aave, and more. These incidents ranged from smart contract vulnerabilities to account compromises and phishing attacks.\n\n🙏 A big thank you to @scamsniffer for their invaluable work in investigating scams.\n\n📖 Read the full report at: https://t.co/Yw3t6JfZ1l'}"
    )

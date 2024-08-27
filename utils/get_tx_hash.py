def get_tx_hash(tx):
    return tx[-66:]


if __name__ == '__main__':
    print(get_tx_hash(
        'https://app.blocksec.com/explorer/tx/optimism/0x6c19762186c9f32c81eb2a79420fc7ad4485aa916cab37ec278b216757bfba0d'))

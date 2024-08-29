import csv

input_file = r'../dataset/raw.csv'
output_file = r'../dataset/utf8Format_standard.csv'

with open(input_file, 'r', encoding='gbk', errors='ignore') as f:
    reader = csv.reader(f)
    next(reader)
    rows = list(reader)

processed_rows = []
for row in rows:
    attack_tx = row[6:-3]
    if len(attack_tx) > 1:
        attack_tx_str = '|'.join(attack_tx)
    else:
        attack_tx_str = attack_tx[0] if attack_tx else ''
    row[6:-3] = [attack_tx_str]
    processed_rows.append(row)
print(processed_rows)


with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(processed_rows)

print("File processed and saved.")

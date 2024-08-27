import csv

input_file = r'data/utf8Format_standard.csv'

t = 'BSC'

line = 1
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    rows = list(reader)
    for row in rows:
        line = line + 1
        chains = row[2]
        tx = row[6]
        if len(chains.split(' ')) != len(tx.split('|')):
            print(f"在csv的第{line}行")
            print(tx.split('|'))
            print(f"有{len(tx.split('|'))}个tx")

input_data = '0xa9059cbb0000000000000000000000008134a2fdc127549480865fb8e5a9e8a8a95a54c50000000000000000000000000000000000000000001cea0eaa7156990ddc40e7'

recipient = "0x" + input_data[34:74]
token_value = int(input_data[74:], 16)

print(recipient)
print(token_value)

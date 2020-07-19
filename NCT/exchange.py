import ccxt

print('Start of the program')
print('-----------------------------')
#
# b = ccxt.binance({
#     'uid': 'noah13nelson@gmail.com',
#     'apiKey': 'ihe8Ac7EWc1Gvme4Pkub00Uml1qdVCYAF2efVYjB3O1RBwm778lzPGYbJm8g3LgL',
#     'secret': 'pGPmiVVLlA6CqhmHyLvS0IGC7roTpUugjnBX2dmQ2rdLjiVdepB2OIiezM9fVkx1',
#     'enableRateLimit': True,
#     'verbose': True
# })
#
# print(b.account())
# print(b.fetch_balance(params={}))
# print(b.fetch_total_balance())
exchange = ccxt.binance({
    'apiKey': 'ihe8Ac7EWc1Gvme4Pkub00Uml1qdVCYAF2efVYjB3O1RBwm778lzPGYbJm8g3LgL',
    'secret': 'pGPmiVVLlA6CqhmHyLvS0IGC7roTpUugjnBX2dmQ2rdLjiVdepB2OIiezM9fVkx1',
    'enableRateLimit': True,
    'verbose': True
})
print(exchange.account())
#print(exchange.fetch_fees())
#print(exchange.fetch_free_balance())
print("Balance************* ",exchange.balance)
print(exchange.fetch_total_balance())
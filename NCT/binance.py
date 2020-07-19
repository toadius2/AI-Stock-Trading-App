import ccxt
import requests

URL_ACCOUNT_INFO = 'https://api.binance.com/api/v3/account'
URL_SERVER_TIME = 'https://api.binance.com/api/v1/time'
URL_YOBIT_INFO = 'https://yobit.net/api'



class ccxt_exchange:

    def __init__(self, username="", password="", exchange="", api_key="", secret=""):

        # exchange : bitstamp, binance, poloniex, yobit, livecoin
        self.exchange_name = exchange
        self.username = username
        self.password = password
        self.api_key = api_key
        self.secret = secret
        self.exchanges = ['bitstamp', 'binance', 'livecoin', 'yobit', 'poloniex']

    def login(self):
        ccxt.binance.account()
        pass

    def get_account_balance(self):
        pass

if __name__ == '__main__':

    # yobit = ccxt.yobit({
    #     'uid': 'noah13nelson@gmail.com',
    #     'apiKey': '8FA4E982CAB11663C11936006F05097B',
    #     'secret': '7ddf527b43a6e876e8dc0a35714bfe2d',
    #     'enableRateLimit': True,
    #     'password': None
    # })
    # print(yobit.has)
    # print('--------------------')
    # print(yobit.fetch_total_balance())

    # ex1 = ccxt.binance()
    # ex1_info = dir(ex1)
    # ex1.account()
    # binance = ccxt.binance({
    #     'uid': 'noah13nelson@gmail.com',
    #     'apiKey': 'oox7xCH0BivEah7LOmfOiHylDEj6tueEArxf3rfj4tgDWFdKNNP9CCNuvKIXxrvR',
    #     'secret': 'ouo5WhOL4ISrmbYJcwANcE9piDHG2KvRmQT0i6AJbqabAGSDsxBeHjKpv7RI1VIa',
    #     'enableRateLimit': True,
    #     'verbose': True,
    #     'password': None
    # })
    # print(binance.fetch_free_balance())
    # print(binance.has)
    # print(binance.fetch_balance())
    # from variable id

    # exchange_id = 'binance'
    # exchange_class = getattr(ccxt, exchange_id)
    # exchange = exchange_class({
    # 'apiKey': 'oox7xCH0BivEah7LOmfOiHylDEj6tueEArxf3rfj4tgDWFdKNNP9CCNuvKIXxrvR',
    # 'secret': 'ouo5WhOL4ISrmbYJcwANcE9piDHG2KvRmQT0i6AJbqabAGSDsxBeHjKpv7RI1VIa',
    # 'timeout': 30000,
    # 'enableRateLimit': True,
    # })

    # print(exchange.fetch_balance())

    # ex2_id = 'yobit'
    # ex2_class = getattr(ccxt, ex2_id)
    # ex2 = ex2_class({
    # 'uid': 'noah13nelson@gmail.com',
    # 'apiKey': '8FA4E982CAB11663C11936006F05097B',
    # 'secret': '7ddf527b43a6e876e8dc0a35714bfe2d',
    # 'enableRateLimit': True,
    # 'password': None
    # })
    #
    # print(ex2.fetch_balance())

    binance = ccxt.binance({
        'uid': 'noah13nelson@gmail.com',
        'apiKey': 'ihe8Ac7EWc1Gvme4Pkub00Uml1qdVCYAF2efVYjB3O1RBwm778lzPGYbJm8g3LgL',
        'secret': 'pGPmiVVLlA6CqhmHyLvS0IGC7roTpUugjnBX2dmQ2rdLjiVdepB2OIiezM9fVkx1',
        'enableRateLimit': True,
    })
    print(binance.fetch_balance())

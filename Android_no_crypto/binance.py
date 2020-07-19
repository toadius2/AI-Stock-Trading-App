import requests
import time
import hmac
import hashlib

base_url = 'https://api.binance.us'

class Binance:
    def __init__(self, apiKey, secretKey):
        ## signature generation required for trades

        self.api_key = apiKey
        self.secret_key = secretKey
        self.session = self.init_session()

    def generate_signature(self, data):
        data_string = '&'.join(['{}={}'.format(key, data[key]) for key in data])
        signature = hmac.new(self.secret_key.encode('utf-8'), data_string.encode('utf-8'), hashlib.sha256)
        return signature.hexdigest()

    def init_session(self):
        session = requests.session()
        session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
        return session

    def place_order(self, direction, currency, curr_quant, ex_currency=''):
        # direction : 'BUY' or 'SELL', will make market transactions
        # currency : Symbol for currency to trade
        # ex_currency : Symbol for currency to trade to or order. default is nothing since currency input will include both symbols
        # curr_quant : Amount to sell or buy

        url = base_url + '/api/v3/order'

        data = {
            'symbol': currency + ex_currency,
            'side': direction,
            'type': 'MARKET',
            'quantity': curr_quant,
            'timestamp': self.synchronize(),
            'recvWindow': 10000
        }

        signature = self.generate_signature(data)
        data['signature'] = signature

        res = self.session.post(url, params=data).json()
        print(res)

    def test_order(self, direction, currency, curr_quant, ex_currency=''):
        url = base_url + '/api/v3/order/test'

        data = {
            'symbol': currency + ex_currency,
            'side': direction,
            'type': 'MARKET',
            'quantity': curr_quant,
            'timestamp': self.synchronize(),
            'recvWindow': 10000
        }

        signature = self.generate_signature(data)
        data['signature'] = signature

        res = self.session.post(url, params=data)
        print(res)
        print(res.json())

    def test_connection(self):
        url = base_url + '/api/v3/ping'

        res = requests.get(url)
        print(res)

    def get_server_time(self):
        url = base_url + '/api/v3/time'
        res = requests.get(url)
        return res.json()

    def exchange_information(self):
        url = base_url + '/api/v3/exchangeInfo'
        res = requests.get(url)
        return res.json()

    def synchronize(self):
        server_time = bin.get_server_time()['serverTime']
        system_time = int(time.time()) * 1000
        offset = system_time - server_time + 500
        return system_time - offset

    def account_information(self):
        url = base_url + '/api/v3/account'
        data = {
            'recvWindow': 10000,
            'timestamp': self.synchronize()
        }
        signature = self.generate_signature(data)
        data['signature'] = signature
        res = self.session.get(url, params=data)
        return res.json()

    def price_ticker(self, symbol=''):
        url = base_url + '/api/v3/ticker/price'
        data = {}
        if not symbol == '':
            data['symbol'] = symbol
        res = requests.get(url, params=data)
        return res.json()

    def get_fit_for_currency(self, symbol):
        balance = 0.0
        for item in self.account_information()['balances']:
            if item['asset'] == 'USDT':
                balance = item['free']
        print("current balance is: ", balance)
        price = self.price_ticker(symbol)['price']

        return float(balance)/float(price)

    def get_currency_in_acc(self, currency):
        # Currency: symbol for currency only, ex: BTC, ETH
        for item in self.account_information()['balances']:
            if item['asset'] == currency:
                return float(item['free'])
        return 0.0

    def get_nonzero_currency(self):
        nonzero = []
        for item in self.account_information()['balances']:
            if float(item['free']) > 0.0:
                nonzero.append(item)
        return nonzero

    def truncate(self, n, decimals=6):
        mult = 10 ** decimals
        return int(n * mult)/mult


# new_api = 'lISh9SeQdCT1HGPeo3Z6p8jWAsOJ6tmjoG7LeMsMNGCGBT0HRhfyfEvHJdjn49IG'
# new_secret = 'mM57MtfNnRG1UrrZs6uGbKNNx1VIU7UktgwqPxelhXkI0cqjXaSCTOvXZY8vMTxj'
# bin = Binance(new_api, new_secret)
# #
# # curr = 'BTC'
# # to_curr = 'USDT'
# # amount = bin.get_fit_for_currency(curr+to_curr)
# print(bin.test_order('SELL', 'USDT', .002, 'BTC'))
# print(amount, acc)
# print(bin.truncate(amount))
# bin.place_order('BUY', curr, bin.truncate(amount), to_curr)

# bin.test_connection()
# ex = bin.exchange_information()
# for arr in ex['symbols']:
#     print(arr['symbol'])
# bin.test_order('BUY', 'BTCUSDT', 0)

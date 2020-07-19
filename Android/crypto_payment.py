import requests
import hmac
import hashlib
import urllib

'''
Public Key: 02e30ffec47be1832b8aaf78f4d44bcba010b0466ebba752729ddfcb46c64f88
Private Key: 9541d1DE244370e23c66F8062A6eD688cc0420890Ff2e14d11F100d05002488b
'''

accepted_coins = ['BTC', 'LTC', 'BCH', 'BNB', 'BSV', 'DASH', 'EOS', 'ETC', 'ETH', 'QTUM', 'RVN',
    'TRX', 'TUSD', 'USDC', 'USDT', 'WAVES', 'XEM', 'XMR', 'XVG', 'ZEC']

test_coin = 'LTCT'

class CryptoPayment:
    def __init__(self):
        self.url = 'https://www.coinpayments.net/api.php'
        self.public_key = '02e30ffec47be1832b8aaf78f4d44bcba010b0466ebba752729ddfcb46c64f88'
        self.private_key = '9541d1DE244370e23c66F8062A6eD688cc0420890Ff2e14d11F100d05002488b'
        self.version = 1

    def create_hmac(self, params):
        encoded = urllib.parse.urlencode(params).encode('utf-8')
        key = hmac.new(bytearray(self.private_key, 'utf-8'), encoded, hashlib.sha512).hexdigest()
        return encoded, key

    def account_info(self):
        params = {
            'cmd': 'get_basic_info',
            'key': self.public_key,
            'version': self.version,
        }
        encoded, key = self.create_hmac(params)
        headers = {
            'HMAC': key
        }

        res = requests.post(self.url, data=params, headers=headers)
        print(res.json())

    def create_transaction(self, curr1, curr2, amount, email):
        params = {
            'cmd': 'create_transaction',
            'key': self.public_key,
            'version': self.version,
            'currency1': curr1,
            'currency2': curr2,
            'amount': amount,
            'buyer_email': email
        }
        encoded, key = self.create_hmac(params)
        headers = {
            'HMAC': key
        }

        res = requests.post(self.url, data=params, headers=headers)
        return res.json()

    def get_tx(self):
        params = {
            'key': self.public_key,
            'version': self.version,
            'cmd': 'get_tx_ids'
        }
        encoded, key = self.create_hmac(params)
        headers = {
            'HMAC': key
        }

        res = requests.post(self.url, data=params, headers=headers)
        return res.json()

    def get_tx_info(self, tx_id):
        params = {
            'key': self.public_key,
            'version': self.version,
            'cmd': 'get_tx_info',
            'txid': tx_id
        }
        encoded, key = self.create_hmac(params)
        headers = {
            'HMAC': key
        }

        res = requests.post(self.url, data=params, headers=headers)
        return res.json()

# cp = CryptoPayment()
# cp.account_info()
# print(cp.create_transaction('USD', 'LTCT', 20, 'tg5849@gmail.com'))
# # CPEB7YXEYBHWUDSIRNKHB8YZWR
# print(cp.get_tx_info('CPEC7RF48UZDGUVXISQ1WAUDEM'))

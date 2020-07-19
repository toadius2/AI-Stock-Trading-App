import requests
import hmac
import hashlib

base_url = 'https://api.bkex.com'

class BKEX:
    def __init__(self, accessKey, secretKey):
        self.access_key = accessKey
        self.secret_key = secretKey
        self.session = requests.session()
        self.session.headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X_ACCESS_KEY': self.access_key,
            'X_SIGNATURE': ''
        }

    def generate_signature(self, data=''):
        if data != '':
            data = sorted(data)
            data_string = '&'.join(['{}={}'.format(key, data[key]) for key in data])
            signature = hmac.new(self.secret_key.encode('utf-8'), data_string.encode('utf-8'), hashlib.sha256)
        else:
            data_string = ''
            signature = hmac.new(self.secret_key.encode('utf-8'), data_string.encode('utf-8'), hashlib.sha256)


        return signature.hexdigest()

    def get_exchange_info(self):
        url = base_url + '/v1/exchangeInfo'

        res = self.session.get(url)
        return res.json()

    def account_information(self):
        url = base_url + '/v1/u/wallet/balance'
        signature = self.generate_signature()
        self.session.headers.update({'X_SIGNATURE': signature})

        res = self.session.get(url)
        return res.json()

    def create_new_order(self, pair, order_price, amount, direction=''):
        #direction: BID or ASK
        url = base_url + '/v1/u/trade/order/create'
        data = {
            'pair': pair,
            'price': order_price,
            'amount': amount
        }
        if direction != '':
            data['direction'] = direction
        signature = self.generate_signature(data=data)
        self.session.headers.update({'X_SIGNATURE': signature})

        res = self.session.post(url, params=data)
        return res.json()



if __name__ == '__main__':
    bkex = BKEX('7deb2a0c2e0801c4e3a24ba7fb2a425a01552c3d120c98b5ce054ba87b53d237',
        'f575bf4218d62f0f328cf61a5a514bbc4f7fca095999133954675d014a2390ac')
    # print(bkex.get_exchange_info())
    # print(bkex.user_account_balance())

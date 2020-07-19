import requests
import time
import hmac
import hashlib

base_url = 'https://www.mxc.ceo'

class MXC:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key

    def generate_signature(self, method, data, path):
        data_string = '&'.join('{}={}'.format(key, data[key]) for key in sorted(data))
        to_sign = '\n'.join([method, path, data_string])
        signature = hmac.new(self.secret_key.encode(), to_sign.encode(), hashlib.sha256)
        return signature.hexdigest()

    def account_info(self):
        url = base_url + '/open/api/v2/account/info'

        data = {
            'api_key': self.api_key,
            'req_time': self.synchronize()
        }

        signature = self.generate_signature('GET', data, url)
        data['sign'] = signature

        res = requests.get(url, params=data)
        return res.json()

    def get_server_time(self):
        url = base_url + '/open/api/v2/common/timestamp'
        res = requests.get(url)
        return res.json()

    def synchronize(self):
        server_time = int(self.get_server_time()['data'])
        system_time = time.time() * 1000
        offset = system_time - server_time
        return system_time - offset + 500

    def all_symbols(self):
        url = base_url + '/open/api/v2/market/symbols'

        data = {
            'api_key': self.api_key,
            'req_time': self.synchronize()
        }

        signature = self.generate_signature('GET', data, url)
        data['sign'] = signature

        return requests.get(url, params=data).json()

if __name__ == '__main__':
    mxc = MXC('mx0hGkMH4gLPb0xAif', '980696eb373b43d3882d3452e0f536d5')
    # print(mxc.server_time()['data'])
    # print(time.time()*1000)
    # k = mxc.all_symbols()['data']
    # for i in k:
    #     print(i['symbol'])
    print(mxc.account_info())

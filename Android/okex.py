import requests
import json
import datetime
import hmac
import base64




#TODO: Add functionalities for Margin Trading, Futures Trading, Perpetual Swap, and Options Trading
# Currently, only Spot Trading is supported.

# http header
API_URL = 'https://www.okex.com'
CONTENT_TYPE = 'Content-Type'
OK_ACCESS_KEY = 'OK-ACCESS-KEY'
OK_ACCESS_SIGN = 'OK-ACCESS-SIGN'
OK_ACCESS_TIMESTAMP = 'OK-ACCESS-TIMESTAMP'
OK_ACCESS_PASSPHRASE = 'OK-ACCESS-PASSPHRASE'

APPLICATION_JSON = 'application/json'

GET = "GET"
POST = "POST"
DELETE = "DELETE"

# account
SPOT_ACCOUNT_INFO = '/api/spot/v3/accounts'
SPOT_ORDER = '/api/spot/v3/orders'



def get_sign(message, secret_key):
    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    return base64.b64encode(d)

def pre_hash(timestamp, method, request_path, body):
    return str(timestamp) + str.upper(method) + request_path + body

def get_header(api_key, sign, timestamp, passphrase):
    header = dict()
    header[CONTENT_TYPE] = APPLICATION_JSON
    header[OK_ACCESS_KEY] = api_key
    header[OK_ACCESS_SIGN] = sign
    header[OK_ACCESS_TIMESTAMP] = str(timestamp)
    header[OK_ACCESS_PASSPHRASE] = passphrase

    return header

def parse_params_to_str(params):
    url = '?'
    for key, value in params.items():
        url = url + str(key) + '=' + str(value) + '&'

    return url[0:-1]

def get_timestamp():
    now = datetime.datetime.utcnow()
    t = now.isoformat("T", "milliseconds")
    return t + "Z"



class Client(object):

    def __init__(self, api_key, api_secret_key, passphrase, first=False):

        self.API_KEY = api_key
        self.API_SECRET_KEY = api_secret_key
        self.PASSPHRASE = passphrase
        self.first = first

    def _request(self, method, request_path, params, cursor=False):
        if method == GET:
            request_path = request_path + parse_params_to_str(params)
        
        # url
        url = API_URL + request_path

        timestamp = get_timestamp()

        body = json.dumps(params) if method == POST else ""
        sign = get_sign(pre_hash(timestamp, method, request_path, str(body)), self.API_SECRET_KEY)
        header = get_header(self.API_KEY, sign, timestamp, self.PASSPHRASE)

        if self.first:
            print("url:", url)
            self.first = False

        print("url:", url)
        print("body:", body)

        # send request
        response = None
        if method == GET:
            response = requests.get(url, headers=header)
        elif method == POST:
            response = requests.post(url, data=body, headers=header)
        elif method == DELETE:
            response = requests.delete(url, headers=header)

        # exception handle
        if not str(response.status_code).startswith('2'):
            raise OkexAPIException(response)
        try:
            res_header = response.headers
            if cursor:
                r = dict()
                try:
                    r['before'] = res_header['OK-BEFORE']
                    r['after'] = res_header['OK-AFTER']
                except:
                    pass
                return response.json(), r
            else:
                return response.json()

        except ValueError:
            raise OkexRequestException('Invalid Response: %s' % response.text)
    
    def _request_without_params(self, method, request_path):
        return self._request(method, request_path, {})
    
    def _request_with_params(self, method, request_path, params, cursor=False):
        return self._request(method, request_path, params, cursor)


class Okex(Client):
    def __init__(self, apiKey, secretKey, pwd, testnet=True):
        """
        OKEx has an environment for testing programs and strategies at Testnet.
        The client connects to testnet by default. 
        To connect to the live environment, set 'testnet' to False.
        """
        Client.__init__(self, apiKey, secretKey, pwd, testnet)

    def account_information(self):
        return self._request_without_params(GET, SPOT_ACCOUNT_INFO)

    def place_order(self, sbl, qty, direction='Buy', prc=''):
        # sbl : Instrument symbol e.g. XBTUSD
        # qty : Amount to sell or buy (positive quantity --> buy, negative quantity --> sell)
        # direction : 'Buy' or 'Sell'
        # prc : Optinal limit price
        params = {'instrument_id': sbl, 'side': direction, 'client_oid': '', 'type': '', 'size': qty, 'price': prc, 'order_type': '0', 'notional': ''}
        return self._request_with_params(POST, SPOT_ORDER, params)


class OkexAPIException(Exception):

    def __init__(self, response):
        self.code = 0
        try:
            json_res = response.json()
        except ValueError:
            self.message = 'Invalid JSON error message from Okex: {}'.format(response.text)
        else:
            if "code" in json_res.keys() and "message" in json_res.keys():
                self.code = json_res['code']
                self.message = json_res['message']
            elif "error_code" in json_res.keys() and "error_message" in json_res.keys():
                self.code = json_res['error_code']
                self.message = json_res['error_message']
            else:
                self.code = 'Please wait a moment'
                self.message = 'Maybe something is wrong'

        self.status_code = response.status_code
        self.response = response
        self.request = getattr(response, 'request', None)

    def __str__(self):  # pragma: no cover
        return 'API Request Error(code=%s): %s' % (self.code, self.message)


class OkexRequestException(Exception):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return 'OkexRequestException: %s' % self.message



if __name__ == '__main__':
    okex = Okex('apiKey', 'secretKey', 'passphrase')
    print(okex.account_information())
    #okex.place_order('TBTC/TUSDT', 1)



# Adapted from: https://github.com/okex/V3-Open-API-SDK/tree/master/okex-python-sdk-api

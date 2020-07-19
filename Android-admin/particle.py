import requests
import threading
import json
import ccxt
from binance import Binance
from Robinhood import Robinhood
from forex import Forex

class Particle:
    def __init__(self, credentials):
        self.headers = {}
        self.secret = '2f7c82f8baca961fbff0e657c7edfb2ad7c9b110'
        self.client_id = 'nct-app-3115'
        # Auth tokens
        self.access_token = ''
        self.refresh_token = ''
        self.thread_run = True
        self.creds = credentials
        self.bin = self.initiate_binance()
        self.forex = Forex()
        self.robin = Robinhood()

    def initiate_binance(self):
        for ex in self.creds['crypto']:
            if ex['exchange'] == 'binance':
                return Binance(ex['apiKey'], ex['secretKey'])
            return None

    def extract_exchanges(self):
        exchanges = []
        crypto_credentials = self.creds['crypto']
        for i in crypto_credentials:
            exchanges.append(i['exchange'])
        return exchanges

    def auth(self):
        url = 'https://api.particle.io/oauth/token'
        payload = {
            'client_id': self.client_id,
            'client_secret': self.secret,
            'grant_type': 'password',
            'password': 'Hittingstride*13',
            'username': 'noah13nelson@gmail.com'
        }
        try:
            res = requests.post(url, data=payload)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(str(e))
            return 'error'

        if 'access_token' in data:
            self.access_token = data['access_token']
            self.refresh_token = data['refresh_token']
            return 'success'
        else:
            return 'failed'

    def get_all_events(self):
        url = "https://api.particle.io/v1/devices/events?access_token={}".format(self.access_token)
        res = requests.get(url, stream=True)
        for data in res.iter_lines():
            print(data)

    def get_events_by_name(self, event_prefix):
        url = 'https://api.particle.io/v1/events/{}?access_token={}'.format(event_prefix, self.access_token)
        res = requests.get(url, stream=True)
        # data in form {'data': 'EXCHANGE CURRENCY TO_CURR DIRECTION %AMOUNT', ...}
        for data in res.iter_lines():
            data = data.decode('utf-8')
            print(data)
            order = ''
            try:
                order = json.loads(data[6:])['data']
                order_split = order.split()
                print(order_split)

                ##########################
                # Crypto Signal Received #
                ##########################

                if order_split[0] == 'CRYPTO':
                    # FORMAT: CRYPTO CURR TO_CURR DIRECTION %AMOUNT
                    if order_split[2] == 'ALL':
                        # FORMAT: CRYPTO SELL ALL
                        # for j in self.extract_exchanges():
                        #     ex = ccxt.eval('ccxt.%s ()' % j)
                        #     # Finish this
                        nonzero = self.bin.get_nonzero_currency()
                        for i in nonzero:
                            try:
                                self.bin.place_order('SELL', i['asset'], self.bin.truncate(amount), 'USDT')
                            except Exception as e:
                                print('Something went wrong')
                    # exchanges = self.discriminate_exchange(order_split[1], order_split[2])

                    percent_port = float(order_split[4])/100
                    direction = order_split[3]
                    to_curr = order_split[2]
                    currency = order_split[1]
                    if direction == 'BUY':
                        amount = self.bin.get_fit_for_currency(currency+to_curr) * percent_port
                    elif direction == "SELL":
                        amount = self.bin.get_currency_in_acc(currency) * percent_port
                    if amount != 0:
                        self.bin.place_order(direction, currency, self.bin.truncate(amount), to_curr)
                    else:
                        print("Amount is 0")

                ##########################
                # Forex Signal Received  #
                ##########################

                elif order_split[0] == 'FOREX':
                    # FORMAT: FOREX CURR TO_CURR DIRECTION %AMOUNT
                    self.forex.login(self.creds['Funame'], self.creds['Fpassw'], self.creds['Fappkey'])
                    percent_port = float(order_split[4])/100
                    direction = order_split[3]
                    to_curr = order_split[2]
                    currency = order_split[1]
                    try:
                        market_id = f.get_marketId_by_name(curr + '/' + to_curr)
                    except Exception as e:
                        market_id = f.get_marketId_by_name(to_curr + '/' + curr)
                    if direction == 'BUY':
                        amount = self.forex.get_margins()['Cash'] * percent_port
                        self.forex.buy_order(market_id, amount, 'buy')
                    if direction == 'SELL':
                        amount = self.forex.get_margins()['Cash'] * percent_port
                        self.forex.buy_order(market_id, amount, 'sell')

                #############################
                # Robinhood signal Received #
                #############################

                elif order_split[0] == 'ROBIN':
                    # FORMAT: ROBIN STOCK_TICKER DIRECTION %AMOUNT
                    for ex in self.creds['stock']:
                        if ex['exchange'] == 'Robinhood':
                            robin_name = ex['username']
                            robin_pass = ex['password']
                    self.robin.login(robin_name, robin_pass)
                    if order_split[2] == 'ALL':
                        # FORMAT: ROBIN SELL all
                        securities = self.robin.securitied_owned()['results']
                        for i in securities:
                            instrument = i['instrument']
                            amount = i['quantity']
                            self.robin.place_market_sell_order(instrument_URL=instrument, time_in_force='GFD', quantity=amount)

                    stock_ticker = order_split[1]
                    direction = order_split[2]
                    percent_port = float(order_split[3])/100
                    if direction == 'BUY':
                        amount = self.robin.get_account()['cash'] * percent_port
                        self.robin.place_market_buy_order(symbol=stock, time_in_force='GFD', quantity=amount)
                    elif direction == 'SELL':
                        instrument = self.robin.intruments('stock')[0]['url']
                        securities = self.robin.securities_owned()
                        for i in securities['results']:
                            if c == i['instrument']:
                                amount = i['quantity'] * percent_port
                            else:
                                amount = None
                                print('Stock not found')
                        if amount <= 0:
                            amount = None
                            print('Illegal amount to trade')
                        self.robin.place_market_sell_order(symbol=stock, time_in_force='GFD', quantity=amount)


                if not self.thread_run:
                    break
            except Exception as e:
                print('particle', e)


    def get_events_from_device(self, device_id, event_prefix=''):
        # event_prefix is optional, following 2 lines prepends '/' if event_prefix is passed as argument
        if event_prefix != '':
            event_prefix = '/' + event_prefix
        url = 'https://api.particle.io/v1/devices/{}/events{}?access_token={}'.format(device_id, event_prefix, self.access_token)
        res = requests.get(url, stream=True)
        for data in res.iter_lines():
            print(data)

    def discriminate_exchange(self, curr, to_curr):
        # 2 combinations since order is different for some exchanges - want to be exhaustive with search
        str_comb1 = curr + '/' + to_curr
        str_comb2 = to_curr + '/' + curr
        exchanges = self.extract_exchanges()
        return_exchanges = []
        for i in exchanges:
            ex = eval('ccxt.%s ()' % i)
            ex.load_markets()
            if str_comb1 in ex.symbols:
                return_exchanges.append(i)
            elif str_comb2 in ex.symbols:
                return_exchanges.append(i)
        if len(return_exchanges) != 0:
            return return_exchanges
        else:
            return 'No valid exchanges found'


    def publish_event(self, event_name, data=''):
        url = 'https://api.particle.io/v1/devices/events'
        payload = {
            'access_token': self.access_token,
            'name': event_name,
            'private': 'true'
        }
        if data != '':
            payload['data'] = data
        try:
            res = requests.post(url, data=payload)
            print(res)
            print(res.json())
        except Exception as e:
            print(str(e))
            return 'error'

        if 'ok' in res:
            if res['ok'] == 'True':
                return 'success'
        else:
            return 'failed'

import ccxt
import time


class Crypto:
    def __init__(self):
        self.apikey = ''
        self.skey = ''
        self.exchange_name = ''


    def gen_account_id(self, apikey, skey, exchange_name):
        self.apikey = apikey
        self.skey = skey
        self.exchange_name = exchange_name

        account_dict = {}
        account_dict[exchange_name] = {}
        account_dict[exchange_name]['apiKey'] = apikey
        account_dict[exchange_name]['secret'] = skey
        account_dict[exchange_name]['options'] = {'adjustForTimeDifference':  True}
        return account_dict


        # ex_id = exchange_name
        # ex_class = getattr(ccxt, ex_id)
        # ex = ex_class({
        #     'apikey': self.apikey,
        #     'secret': self.skey,
        #     'options': {'adjustForTimeDifference': True}
        # })
        #
        # return ex

    def get_account_balance(self, acc):
        balance = {}

        try:
            exchange = next(iter(acc))
            ex_class = getattr(ccxt, exchange)
            acc_info = ex_class(acc[exchange])
            info = acc_info.fetch_free_balance()
            for i in info.keys():
                if info[i] > .1:
                    balance[i] = info[i]

        except Exception as e:
            print(str(e))
            return 'error'

        return balance

    def sell_order_market(self, acc, market_id, amount):
        exchange = next(iter(acc))
        ex_class = getattr(ccxt, exchange)
        acc_info = ex_class(acc[exchange])
        try:
            place_order = acc_info.create_market_sell_order(market_id, amount)
        except Exception as e:
            print(str(e))
            return 'error'

        return place_order

    def buy_order_market(self, acc, market_id, amount):
        exchange = next(iter(acc))
        ex_class = getattr(ccxt, exchange)
        acc_info = ex_class(acc[exchange])
        try:
            place_order = acc_info.create_market_buy_order(market_id, amount)
        except Exception as e:
            print(str(e))
            return 'error'

        return place_order

    def sell_order_limit(self, acc, market_id, amount, price):
        exchange = next(iter(acc))
        ex_class = getattr(ccxt, exchange)
        acc_info = ex_class(acc[exchange])
        try:
            place_order = acc_info.create_limit_sell_order(market_id, amount, price)
        except Exception as e:
            print(str(e))
            return 'error'

        return place_order

    def buy_order_limit(self, acc, market_id, amount, price):
        exchange = next(iter(acc))
        ex_class = getattr(ccxt, exchange)
        acc_info = ex_class(acc[exchange])
        try:
            place_order = acc_info.create_limit_buy_order(market_id, amount, price)
        except Exception as e:
            print(str(e))
            return 'error'

        return place_order

if __name__ == "__main__":
    c = Crypto()
    id = c.gen_account_id('lISh9SeQdCT1HGPeo3Z6p8jWAsOJ6tmjoG7LeMsMNGCGBT0HRhfyfEvHJdjn49IG',
        'mM57MtfNnRG1UrrZs6uGbKNNx1VIU7UktgwqPxelhXkI0cqjXaSCTOvXZY8vMTxj', 'binance')
    markets = c.get_account_balance(id)

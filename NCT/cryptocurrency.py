import ccxt
import time

def get_exchange_key_dict(apikey, skey, exchange_name):
    '''
    Creates a dict for the account keys
    :param apikey: account API Key
    :param skey: account Secret Key
    :param exchange_name: name of the exchange, example "Binance", "Kraken" etc
    :return: dictionary with api, secret and other mandatory options
    '''
    account_dict = {}
    account_dict[exchange_name] = {}
    account_dict[exchange_name]['apiKey'] = apikey
    account_dict[exchange_name]['secret'] = skey
    account_dict[exchange_name]['options'] = {'adjustForTimeDifference':  True}
    return account_dict


accounts1 = get_exchange_key_dict("ihe8Ac7EWc1Gvme4Pkub00Uml1qdVCYAF2efVYjB3O1RBwm778lzPGYbJm8g3LgL",
                                  "pGPmiVVLlA6CqhmHyLvS0IGC7roTpUugjnBX2dmQ2rdLjiVdepB2OIiezM9fVkx1","binance")

def get_account_balance(accounts, exchange):
    '''
    Check the balance of an account in an exchange
    :param account: list of accounts above
    :param exchange: exchange with the account we are checking
    :return: information and balance in a tuple structure.
    '''
    balance = 0
    e = ""

    try:
        info = getattr(ccxt, exchange)
        info = info(accounts[exchange])
        info.load_markets()
        info = info.fetch_free_balance()
        for i in info.values():
            balance = balance + i
    except Exception:
        e = "authentication error"
        print(exchange, 'authentication error')

    return balance, e

def get_all_exchnages():
    '''
    :return: returns all the exchanges in ccxt
    '''
    all_exchanges = ccxt.exchanges()
    return all_exchanges

def buy_sell_market(self, account, symbol, side, amount, stop_loss_price=None):
    FEE_RATE = 0.0025  # TODO: improve fee determination
    status = "Pending"
    open_ts = time.time()
    ccxt_exchange = account.get_ccxt_exchange()
    ccxt_market = ccxt_exchange.market(symbol)
# ​    amount_prec = ccxt_market["precision"]["amount"]
#     price_prec = ccxt_market["precision"]["price"]
#     amount = round(amount, amount_prec)
#     stop_loss_price = round(stop_loss_price, price_prec)
#     fee = round(amount * FEE_RATE, price_prec)
#     base_change = 0
#     quote_change = 0
#
#     if side == "buy":
#         response = ccxt_exchange.create_market_buy_order(symbol, amount)
#         order_id = response["id"]
#         base_change = amount
#         quote_change = -amount - fee
#     elif side == "sell":
#         response = ccxt_exchange.create_market_sell_order(symbol, amount)
#         order_id = response["id"]
#         base_change = -amount
#         quote_change = amount - fee
#         open_sell_revenue = quote_change
#     status = "Pending"
#
#     return (base_change, quote_change)

if __name__ == '__main__':
    # crypto_balance, crypto_auth = get_account_balance(get_exchange_key_dict(crypto_apikey, crypto_secretekey, crypto_exchange), crypto_exchange))
    apikey = "ihe8Ac7EWc1Gvme4Pkub00Uml1qdVCYAF2efVYjB3O1RBwm778lzPGYbJm8g3LgL"
    secretkey = "pGPmiVVLlA6CqhmHyLvS0IGC7roTpUugjnBX2dmQ2rdLjiVdepB2OIiezM9fVkx1"

    print(get_account_balance(get_exchange_key_dict(apikey, secretkey, "binance"), "binance"))


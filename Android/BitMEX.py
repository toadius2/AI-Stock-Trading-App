import bitmex


class Bitmex:
    def __init__(self, apiKey, secretKey, testnet=True):
        """
        BitMEX has an environment for testing programs and strategies at Testnet.
        The client connects to testnet by default. 
        To connect to the live environment, set 'testnet' to False.
        """
        self.acc = bitmex.bitmex(
            test=testnet, 
            api_key=apiKey, 
            api_secret=secretKey)

    def account_information(self):
        return self.acc.User.User_getWallet().result()

    def place_order(self, sbl, qty, direction='Buy', prc=None):
        # sbl : Instrument symbol e.g. XBTUSD
        # qty : Amount to sell or buy (positive quantity --> buy, negative quantity --> sell)
        # direction : 'Buy' or 'Sell'
        # prc : Optinal limit price
        print(self.acc.Order.Order_new(symbol=sbl, orderQty=qty, side=direction, price=prc).result())


if __name__ == '__main__':
    exchange = Bitmex('apiKey', 'secretKey')
    print(exchange.account_information())
    #exchange.place_order('XBTUSD', 1)



# Reference: https://github.com/BitMEX/api-connectors/tree/master/official-http/python-swaggerpy
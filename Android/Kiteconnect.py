from kiteconnect import KiteConnect
import logging


class Kiteconnect:
    def __init__(self, API_KEY, REQUEST_TOKEN, API_SECRET):
        self.kite = KiteConnect(api_key=API_KEY)
        self.data = self.kite.generate_session(
            REQUEST_TOKEN, api_secret=API_SECRET)
        self.kite.set_access_token(self.data["access_token"])
        self.exchange = None
        self.transaction_type = None

    def postOrders(self, side, market, stockCode, quantity, order_type, product, variety):
        if market == "BSE":
            self.exchange = self.kite.EXCHANGE_BSE
        elif market == "NSE":
            self.exchange = self.kite.EXCHANGE_NSE

        if side == "buy":
            self.transaction_type = self.kite.TRANSACTION_TYPE_BUY
        elif side == "sell":
            self.transaction_type = self.kite.TRANSACTION_TYPE_SELL

        logging.basicConfig(level=logging.DEBUG)
        try:
            order_id = self.kite.place_order(tradingsymbol=stockCode,
                                             exchange=self.kite.EXCHANGE_BSE,
                                             transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                                             quantity=quantity,
                                             order_type=order_type,
                                             product=product,
                                             variety=variety)

            logging.info("Order placed. ID is: {}".format(order_id))
        except Exception as e:
            logging.info("Order placement failed: {}".format(str(e)))

import requests
import json

# Replace demo endpoints with live endpoints while deployment

# Trading endpoint (live) = "https://api.alpaca.markets"


class Alpaca:
    def __init__(self, API_KEY, SECRET_KEY):
        self.TRADING_ENDPOINT = "https://paper-api.alpaca.markets"

        self.ACCOUNT_URL = "{}/v2/account".format(self.TRADING_ENDPOINT)
        self.ORDERS_URL = "{}/v2/orders".format(self.TRADING_ENDPOINT)
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.HEADERS = {'APCA-API-KEY-ID': API_KEY,
                        'APCA-API-SECRET-KEY': SECRET_KEY}

    def postOrders(self, symbol, qty, side, type, time_in_force):

        data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force

        }

        r = requests.post(self.ORDERS_URL, json=data, headers=self.HEADERS)
        return json.loads(r.content)


if __name__ == '__main__':
    API_KEY = "PKDORC9AVWCD76ORX9G0"
    SECRET_KEY = "uPmNGZWXp4P6lCgZC0TAMhXWmLuV8J/IPWddKPzD"
    alpaca = Alpaca(API_KEY, SECRET_KEY)
    response = alpaca.postOrders("HTZ", 50,  "buy", "market", "gtc")
    print(response)

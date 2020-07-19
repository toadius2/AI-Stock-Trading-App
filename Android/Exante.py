from requests.auth import HTTPBasicAuth
import requests
import json

# Replace demo endpoints with live endpoints while deployment

# Data endpoint (live) = "https://api-live.exante.eu/md/"
# Trading endpoint (live) = "https://api-live.exante.eu/trade/"


class Exante:
    def __init__(self, APPLICATION_ID, ACCESS_KEY):
        self.DATA_ENDPOINT = "https://api-demo.exante.eu/md/"
        self.TRADING_ENDPOINT = "https://api-demo.exante.eu/trade/"
        self.ORDERS_ENDPOINT = "{}1.0/orders".format(self.TRADING_ENDPOINT)
        self.APPLICATION_ID = APPLICATION_ID
        self.ACCESS_KEY = ACCESS_KEY

        login_request = requests.get(
            "{}1.0/accounts".format(self.DATA_ENDPOINT), auth=(APPLICATION_ID, ACCESS_KEY))
        self.accountId = json.loads(login_request.content)[0]['accountId']

    def postOrders(self, side, symbol, marketCode, qty, orderType, duration):
        data = {
            "account": self.accountId,
            "instrument": symbol+"."+marketCode,
            "side": side,
            "quantity": qty,
            "orderType": orderType,
            "duration": duration
        }
        r = requests.post(self.ORDERS_ENDPOINT, json=data, auth=(
            APPLICATION_ID, ACCESS_KEY))
        print(json.loads(r.content))

# Parameters to test code


side = "buy"
marketCode = "SGX"
symbol = "5CP"
qty = "1"
orderType = "market"
duration = "day"
APPLICATION_ID = "2e97ef4d-ad68-46dd-ba17-3851a857c0c6"
ACCESS_KEY = "6ZxHHjl8qnRWscZubtSU"

# Function call
exante = Exante(APPLICATION_ID, ACCESS_KEY)
exante.postOrders(side, symbol, marketCode, qty, orderType, duration)

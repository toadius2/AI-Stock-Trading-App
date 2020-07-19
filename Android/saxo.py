from saxo_openapi import API
import saxo_openapi.endpoints.trading as tr
import saxo_openapi.endpoints.portfolio as pf
from saxo_openapi.contrib.orders import tie_account_to_order, MarketOrderFxSpot
from saxo_openapi.contrib.session import account_info
import json
class Saxo:
    def __init__(self, token):
        self.token = token
        self.client = API(access_token=self.token)
    def checkPosition(self):
        r = pf.positions.PositionsMe()
        rv = self.client.request(r)
        print(json.dumps(rv, indent=2))
    def postOrders(self, uic, assetType, side, exchangeId, symbol, qty, orderType, orderRelation, duration, accountkey):
        MO = [
            {
                "Uic": uic,
                "AssetType": assetType,
                "BuySell": side,
                "ExchangeId": exchangeId,
                "Symbol": symbol,
                "Amount": qty,
                "OrderType": orderType,
                "OrderRelation": orderRelation,
                "ManualOrder": False,
                "OrderDuration": {
                    "DurationType": duration
                },
                "AccountKey": accountkey,
            }
        ]
        for r in [tr.orders.Order(data=orderspec) for orderspec in MO]:
            self.client.request(r)
tok = "eyJhbGciOiJFUzI1NiIsIng1dCI6IjhGQzE5Qjc0MzFCNjNFNTVCNjc0M0QwQTc5MjMzNjZCREZGOEI4NTAifQ.eyJvYWEiOiI3Nzc3NyIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiVGx0OUNjYXBvRUo3WEE4VkxzblFkdz09IiwiY2lkIjoiVGx0OUNjYXBvRUo3WEE4VkxzblFkdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiYmY0MjFjMzZmN2FiNDg4MDg0ZGFmMzI5NThmNWY4ZGIiLCJkZ2kiOiI4NCIsImV4cCI6IjE1OTI1MDE5NDUifQ.SulHzbfdL7cE-fqOnYpSTDUpPkMRJKruQ1FuL_OpZ75MCXQB2tsB2CyEpgt-RSCq49GKxRZXpXxTuvWEjwNx5g"
uic = "211"
assetType = "Stock"
side = "Buy"
exchangeId = "NASDAQ"
symbol = "AAPL:xnas"
qty = 1
orderType = "Market"
orderRelation = "StandAlone"
duration = "DayOrder"
accountkey = "Tlt9CcapoEJ7XA8VLsnQdw=="
if __name__ == '__main__':
    saxo = Saxo(tok)
    saxo.checkPosition()
    saxo.postOrders(uic, assetType, side, exchangeId, symbol,
                    qty, orderType, orderRelation, duration, accountkey)
    saxo.checkPosition()
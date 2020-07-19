# All Forex endpoints/URLs

def login():
    return "https://ciapi.cityindex.com/TradingAPI/session"

#{'Session': 'cc229922-813a-4489-9d8a-e73f60514cb3', 'PasswordChangeRequired': False,
# 'AllowedAccountOperator': False, 'StatusCode': 1, 'AdditionalInfo': None,
# 'Is2FAEnabled': False, 'TwoFAToken': None,
# 'Additional2FAMethods': None,
# 'UserType': 1}

# {
#     "AdditionalInfo": null,
#     "StatusCode": 0,
#     "HttpStatus": 401,
#     "ErrorMessage": "Sorry, your login has failed|This is because your login credentials do not match our records.",
#     "ErrorCode": 4010
# }
def get_user_info():
    return "https://ciapi.cityindex.com/TradingAPI/UserAccount/ClientAndTradingAccount"

#     The correct response looks like this:
#     {
#     "LogonUserName": "Mr Noah Nelson - Test",
#     "ClientAccountId": 401589756,
#     "CultureId": 9,
#     "ClientAccountCurrency": "USD",
#     "AccountOperatorId": 400803199,
#     "TradingAccounts": [
#         {
#             "TradingAccountId": 402043148,
#             "TradingAccountCode": "DA545354",
#             "TradingAccountStatus": "Open",
#             "TradingAccountType": "CFD"
#         }
#     ],
#     "PersonalEmailAddress": "noah13nelson@gmail.com",
#     "HasMultipleEmailAddresses": false,
#     "AccountHolders": [
#         {
#             "LegalPartyId": 403205905,
#             "Name": "Mr Noah Nelson - Test"
#         }
#     ],
#     "ClientTypeId": 1,
#     "LinkedClientAccounts": [],
#     "IsNfaEnabledClient": true,
#     "IsFifo": null,
#     "DaysUntilExpiryForDemo": 3.028748418794,
#     "LegalPartyUniqueReference": 6859124,
#     "Is2FAEnabledAO": false,
#     "Regulatory": {
#         "IsMiFIDRegulator": false,
#         "IsPiisProvided": false,
#         "ClientAccountCreationDate": "/Date(1559865600000)/"
#     },
#     "IsDMAClient": false
# }
   # The invalid session response
# {
#     "HttpStatus": 401,
#     "ErrorMessage": "Session is not valid",
#     "ErrorCode": 4011
# }

def get_client_account_margin():
    return "https://ciapi.cityindex.com/TradingAPI/margin/ClientAccountMargin"

    # The Sample response:
    # {
    #     "Cash": 49999.02,
    #     "Margin": 0,
    #     "MarginIndicator": -1,
    #     "NetEquity": 49999.02,
    #     "OpenTradeEquity": 0,
    #     "TradableFunds": 49999.02,
    #     "PendingFunds": 0,
    #     "TradingResource": 49999.02,
    #     "TotalMarginRequirement": 0,
    #     "CurrencyId": 11,
    #     "CurrencyIsoCode": "USD"
    # }

def validate_session():
    return "https://ciapi.cityindex.com/TradingAPI/session/validate"
# Response for this API call looks as follows
# {
#     "IsAuthenticated": false
# }

#### TradeHistory API call response
# {
#     "TradeHistory": [
#         {
#             "OrderId": 687129114,
#             "OpeningOrderIds": [
#                 687129114
#             ],
#             "MarketId": 401484392,
#             "MarketName": "GBP/USD",
#             "Direction": "buy",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 1.27063,
#             "TradingAccountId": 402043148,
#             "Currency": "USD",
#             "RealisedPnl": null,
#             "RealisedPnlCurrency": "",
#             "LastChangedDateTimeUtc": "/Date(1561484358417)/",
#             "ExecutedDateTimeUtc": "/Date(1561484338043)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.13,
#             "Commission": null
#         },
#         {
#             "OrderId": 687129220,
#             "OpeningOrderIds": [
#                 687129114
#             ],
#             "MarketId": 401484392,
#             "MarketName": "GBP/USD",
#             "Direction": "sell",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 1.27023,
#             "TradingAccountId": 402043148,
#             "Currency": "USD",
#             "RealisedPnl": -0.4,
#             "RealisedPnlCurrency": "USD",
#             "LastChangedDateTimeUtc": "/Date(1561484358417)/",
#             "ExecutedDateTimeUtc": "/Date(1561484358417)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.12,
#             "Commission": null
#         },
#         {
#             "OrderId": 687133711,
#             "OpeningOrderIds": [
#                 687133711
#             ],
#             "MarketId": 401484414,
#             "MarketName": "USD/JPY",
#             "Direction": "buy",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 107.17,
#             "TradingAccountId": 402043148,
#             "Currency": "JPY",
#             "RealisedPnl": null,
#             "RealisedPnlCurrency": "",
#             "LastChangedDateTimeUtc": "/Date(1561486117253)/",
#             "ExecutedDateTimeUtc": "/Date(1561485819777)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.07,
#             "Commission": null
#         },
#         {
#             "OrderId": 687134452,
#             "OpeningOrderIds": [
#                 687133711
#             ],
#             "MarketId": 401484414,
#             "MarketName": "USD/JPY",
#             "Direction": "sell",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 107.129,
#             "TradingAccountId": 402043148,
#             "Currency": "JPY",
#             "RealisedPnl": -0.38,
#             "RealisedPnlCurrency": "USD",
#             "LastChangedDateTimeUtc": "/Date(1561486117253)/",
#             "ExecutedDateTimeUtc": "/Date(1561486117253)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.08,
#             "Commission": null
#         },
#         {
#             "OrderId": 687153188,
#             "OpeningOrderIds": [
#                 687153188
#             ],
#             "MarketId": 401484414,
#             "MarketName": "USD/JPY",
#             "Direction": "buy",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 107.166,
#             "TradingAccountId": 402043148,
#             "Currency": "JPY",
#             "RealisedPnl": null,
#             "RealisedPnlCurrency": "",
#             "LastChangedDateTimeUtc": "/Date(1561498986677)/",
#             "ExecutedDateTimeUtc": "/Date(1561498966750)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.1,
#             "Commission": null
#         },
#         {
#             "OrderId": 687153202,
#             "OpeningOrderIds": [
#                 687153188
#             ],
#             "MarketId": 401484414,
#             "MarketName": "USD/JPY",
#             "Direction": "sell",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 107.14,
#             "TradingAccountId": 402043148,
#             "Currency": "JPY",
#             "RealisedPnl": -0.24,
#             "RealisedPnlCurrency": "USD",
#             "LastChangedDateTimeUtc": "/Date(1561498986677)/",
#             "ExecutedDateTimeUtc": "/Date(1561498986677)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.14,
#             "Commission": null
#         },
#         {
#             "OrderId": 689260286,
#             "OpeningOrderIds": [
#                 689260286
#             ],
#             "MarketId": 401484317,
#             "MarketName": "AUD/USD",
#             "Direction": "buy",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 0.70174,
#             "TradingAccountId": 402043148,
#             "Currency": "USD",
#             "RealisedPnl": null,
#             "RealisedPnlCurrency": "",
#             "LastChangedDateTimeUtc": "/Date(1563473389840)/",
#             "ExecutedDateTimeUtc": "/Date(1563300493927)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.1,
#             "Commission": null
#         },
#         {
#             "OrderId": 689541962,
#             "OpeningOrderIds": [
#                 689541962
#             ],
#             "MarketId": 401484335,
#             "MarketName": "EUR/GBP",
#             "Direction": "buy",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 0.89943,
#             "TradingAccountId": 402043148,
#             "Currency": "GBP",
#             "RealisedPnl": null,
#             "RealisedPnlCurrency": "",
#             "LastChangedDateTimeUtc": "/Date(1563474235147)/",
#             "ExecutedDateTimeUtc": "/Date(1563473279893)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.09,
#             "Commission": null
#         },
#         {
#             "OrderId": 689542113,
#             "OpeningOrderIds": [
#                 689260286
#             ],
#             "MarketId": 401484317,
#             "MarketName": "AUD/USD",
#             "Direction": "sell",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 0.70416,
#             "TradingAccountId": 402043148,
#             "Currency": "USD",
#             "RealisedPnl": 2.42,
#             "RealisedPnlCurrency": "USD",
#             "LastChangedDateTimeUtc": "/Date(1563473389840)/",
#             "ExecutedDateTimeUtc": "/Date(1563473389840)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.11,
#             "Commission": null
#         },
#         {
#             "OrderId": 689546055,
#             "OpeningOrderIds": [
#                 689541962
#             ],
#             "MarketId": 401484335,
#             "MarketName": "EUR/GBP",
#             "Direction": "sell",
#             "OriginalQuantity": 1000,
#             "Quantity": 0,
#             "Price": 0.89941,
#             "TradingAccountId": 402043148,
#             "Currency": "GBP",
#             "RealisedPnl": -0.03,
#             "RealisedPnlCurrency": "USD",
#             "LastChangedDateTimeUtc": "/Date(1563474235147)/",
#             "ExecutedDateTimeUtc": "/Date(1563474235147)/",
#             "TradeReference": null,
#             "ManagedTrades": [],
#             "OrderReference": null,
#             "Source": "G2",
#             "IsCloseBy": false,
#             "Liquidation": false,
#             "FixedInitalMargin": 0,
#             "SpreadCost": -0.11,
#             "Commission": null
#         }
#     ],
#     "SupplementalOpenOrders": []
# }

# ****** Sample resposne of Sell_order ***************************************
# {
# 'Status':1,
# 'StatusReason':1,
# 'OrderId':690460273,
# 'Orders':[
# {
# 'OrderId':690460273,
# 'StatusReason':1,
# 'Status':3,
# 'OrderTypeId':1,
# 'Price':0.9107,
# 'Quantity':1000.0,
# 'TriggerPrice':0.0,
# 'CommissionCharge':0.0,
# 'IfDone':[
# ],
# 'GuaranteedPremium':0.0,
# 'OCO':None,
# 'AssociatedOrders':{
# 'Stop':None,
# 'Limit':None
# },
# 'Associated':False
# }
# ],
# 'Quote':None,
# 'Actions':[
# {
# 'ActionedOrderId':0,
# 'ActioningOrderId':0,
# 'Quantity':1000.0,
# 'ProfitAndLoss':0.0,
# 'ProfitAndLossCurrency':None,
# 'OrderActionTypeId':1
# }
# ],
# 'ErrorMessage':None
# }{'LoggedOut': True}
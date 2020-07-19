import json
import requests

URL_LOGIN = "https://ciapi.cityindex.com/TradingAPI/session"
URL_GET_USER_INFO = "https://ciapi.cityindex.com/TradingAPI/useraccount/ClientAndTradingAccount"
URL_GET_CLIENT_ACCOUNT_MARGIN = "https://ciapi.cityindex.com/TradingAPI/margin/ClientAccountMargin"
URL_VALIDATE_SESSION = "https://ciapi.cityindex.com/TradingAPI/session/validate"
URL_LOGOFF = "https://ciapi.cityindex.com/TradingAPI/session/deleteSession"
URL_BUY_SELL = "https://ciapi.cityindex.com/TradingAPI/order/newtradeorder"
URL_SEARCH_BY_MARKET_NAME = "https://ciapi.cityindex.com/TradingAPI/market/search"
URL_PRICE_BAR_HISTORY = "https://ciapi.cityindex.com/TradingAPI/market/{}/barhistory"
URL_LIST_CLIENT_TRADE_HISTORY = "https://ciapi.cityindex.com/TradingAPI/order/tradehistory"

# List of possible Forex server request/response keys
SESSION = "Session"
USERNAME = "UserName"
CASH = "Cash"
MARGIN = "Margin"
CURRENCY = "Currency"
CURRENCY_ISO_CODE = "CurrencyIsoCode"
IS_ERROR = "IsError"
CLIENT_ACCOUNT_ID = "ClientAccountId"
TRADING_ACCOUNT_ID = "TradingAccountId"
ACCOUNT_OPERATOR_ID = "AccountOperatorId"
TRADING_ACCOUNTS = "TradingAccounts"
HTTP_STATUS = "HttpStatus"
IS_2FA_ENABLED = "Is2FAEnabled"
LOGGED_OUT = "LoggedOut"


class Forex:
    def __init__(self):
        self.uname = ''
        self.passw = ''
        self.app_key = ''
        self.client_acc_id = ''
        self.trading_acc_id = ''
        self.session = requests.session()
        self.auth_token = None
        # self.is_twoFA_enabled = None  *2FA is Currently not supported in Forex.com
        self.headers = {
            "Content-Type": "application/json",
        }
        self.session.headers = self.headers

    def login(self, uname, passw, app_key):
        self.uname = uname
        self.passw = passw
        self.app_key = app_key
        print('logging in')
        if self.uname == '' or self.passw == '':
            return False
        request_body = {
            "UserName": self.uname,
            "Password": self.passw,
            "AppVersion": '1',
            "AppComments": '',
            "AppKey": self.app_key
        }

        try:
            result = requests.post(URL_LOGIN, json=request_body)
            result_data = result.json()
        except Exception as e:
            print(str(e))
            return False

        if SESSION in result_data.keys():
            self.auth_token = result_data[SESSION]
            return True

    def user_info(self):
        if self.auth_token is None:
            print('not logged in')
        if not self.validate_session():
            print('invalid session. Login again')
        url = "https://ciapi.cityindex.com/TradingAPI/useraccount/ClientAndTradingAccount?Session={}&UserName={}".format(self.auth_token, self.uname)
        try:
            result = requests.get(url)
            result_data = result.json()
        except Exception as e:
            print("error " + str(e))

        self.client_acc_id = result_data[CLIENT_ACCOUNT_ID]
        return{
            CLIENT_ACCOUNT_ID: result_data[CLIENT_ACCOUNT_ID],
            TRADING_ACCOUNT_ID: result_data[TRADING_ACCOUNTS][0][TRADING_ACCOUNT_ID],
            ACCOUNT_OPERATOR_ID: result_data[ACCOUNT_OPERATOR_ID]
        }

    def get_margins(self):
        if self.auth_token is None:
            print('not logged in')
        if not self.validate_session():
            print('invalid session. Login again')
        query = {SESSION: self.auth_token, USERNAME: self.uname}
        url = "https://ciapi.cityindex.com/TradingAPI/margin/ClientAccountMargin?Session={}&UserName={}".format(self.auth_token, self.uname)
        try:
            result = requests.get(url)
            result_data = result.json()
        except Exception as e:
            print(str(e))

        return{
            CASH: result_data[CASH],
            'NetEquity': result_data['NetEquity'],
            'TradableFunds': result_data['TradableFunds'],
            MARGIN: result_data[MARGIN],
            CURRENCY: result_data[CURRENCY_ISO_CODE]
        }

    def logoff(self):
        print(self.auth_token)
        request = {
            SESSION: self.auth_token,
            USERNAME: self.uname
        }
        url = 'https://ciapi.cityindex.com/TradingAPI/session/deleteSession?userName={}&session={}'.format(self.uname, self.auth_token)
        try:
            result = requests.post(url, json=request)
            result_data = result.json()
            print(result_data)
        except Exception as e:
            print(str(e))

        if result_data[LOGGED_OUT]:
            return True
        return False

    def validate_session(self):
        request_body = {
            SESSION: self.auth_token,
            USERNAME: self.uname
        }
        try:
            result = requests.post(URL_VALIDATE_SESSION, json=request_body)
            result_data = result.json()
            print(result_data)
        except Exception as e:
            print(str(e))

        if 'IsAuthenticated' in result_data.keys() and result_data['IsAuthenticated'] is True:
            return True

        return False

    def market_search(self):
        url = 'https://ciapi.cityindex.com/TradingAPI/market/fullsearchwithtags'
        self.session.headers['Session'] = self.auth_token
        self.session.headers['UserName'] = self.uname
        res = self.session.get(url, params={'tagId': 146})
        return res.json()

    def market_information(self, market_id):
        url = 'https://ciapi.cityindex.com/TradingAPI/market/{}/information'.format(market_id)
        res = self.session.get(url)
        return res.json()

    def list_open_positions(self):
        id = self.user_info()[TRADING_ACCOUNT_ID]
        url = "https://ciapi.cityindex.com/openpositions"
        res = requests.get(url, params = {'TradingAccountId': id})
        print(res)

    def buy_order(self, market_id, quan, direction):
        """
        Makes an API call to place a new BUY order
        :param market_id: The Id of the market to buy
        :param quan: Quantity eg. 1000
        :return: For 200 status code, returns Order Id (eg. 687133711) of the new order placed
                 otherwise None. (Such an order is called as Open position)
        """
        trading_accout_id = self.user_info()[TRADING_ACCOUNT_ID]
        # Get current rate of this market
        rate = self.get_current_rate(market_id)
        if rate is None:
            print("Error occured in Get market rate!")
            return None

        null = None
        false = False
        true = True

        request_body = {
            # "OcoOrder": null,
            # "Type":null,
            # "LastChangedDateTimeUTCDate": null,
            # "ExpiryDateTimeUTC": null,
            # "Applicability": null,
            "Direction": direction,
            # "ExpiryDateTimeUTCDate": null,
            # "TriggerPrice": null,
            "BidPrice": rate,
            # "AuditId": "8049808-0-0-0-R",
            "AutoRollover": false,
            "MarketId": market_id,
            "isTrade": true,
            "OfferPrice": rate,
            "OrderId": 0,
            # "LastChangedDateTimeUTC": null,
            # "Currency": null,
            "Quantity": quan,
            # "QuoteId": null,
            "TradingAccountId": trading_accout_id, #402043148,
            #"MarketName": market_name,
            "PositionMethodId": 1,
            "Status": null,
            "IfDone": []
        }

        parameters = {SESSION: self.auth_token, USERNAME: self.uname}

        try:
            res = requests.post(URL_BUY_SELL, json=request_body, params=parameters)
            res_data_json = res.json()
            print("Buy order data************\n", res_data_json)

        except requests.exceptions.HTTPError as e:
            raise requests.exceptions.HTTPError(e.strerror)

        if res.status_code == 200:
            print("Trade Order successful, OrderId is", res_data_json['OrderId'])
            return res_data_json['OrderId']

        return res_data_json['OrderId']

    def sell_order(self, market_id, order_id, quan):
        """
        Makes an api call to place a Sell order, also called as Closing an open position
        :param market_id: Market id to sell
        :param order_id: The corresponding Open position's id or Order id or can be a list of order_ids's
        :param quan: quantity to be sold like 1000
        :return: For status code 200, returns res_data_json['orderid'] with keys Status, StatusReason, OrderId, Price, Qunatity...
                otherwise returns None
        """
        close_ids = [order_id]
        rate = self.get_current_rate(market_id)

        request_body = {
            "ifDone": [],
            "marketId": market_id,
            "direction": "sell",
            "quantity": quan,
            "bidPrice": rate,
            "close": close_ids,
            "offerPrice": rate,
            "orderId": 0,
            "tradingAccountId": 402043148
        }

        parameters = {SESSION: self.auth_token, USERNAME: self.uname}
        try:
            res = requests.post(URL_BUY_SELL, json=request_body, params=parameters)
            res_data_json = res.json()
            print(res_data_json)

        except requests.exceptions.HTTPError as e:
            raise requests.exceptions.HTTPError(e.strerror)

        if res.status_code == 200:
            print("Trade Sell Order successful!")
            return res_data_json['OrderId']

        return None

    def get_current_rate(self, market_id, price_type="MID"):
        '''
        Makes an API call to get the latest price bar of market_id
        :param market_id: market for which current price is to be fetched
        :param price_type: can have values: ASK(Offer price), BID or MID
        :return: price
        '''
        parameters = {
            "MarketId": market_id,
            "interval": "minute",
            "span": 1,
            "PriceBars": 20,
            "PriceType": price_type,
            # PriceType=BID / ASK / MID
            SESSION: self.auth_token,
            USERNAME: self.uname
        }

        try:
            res = requests.get(URL_PRICE_BAR_HISTORY.format(market_id), params=parameters)
            res_data_json = res.json()
            print("Market rate ********/n",res_data_json)

        except requests.exceptions.HTTPError as e:
            raise requests.exceptions.HTTPError(e.strerror)

        if res.status_code == 200:
            # most recent data point
            data = res_data_json['PriceBars'][-1]
            data_close = data['Close']
            return data_close

        return None


    def get_marketId_by_name(self, market_name):
        '''
        Makes an api call to get market-id for the given market name
        :param market_name: market_name format should be - "EUR/USD", "GBP/AUD"
        :return: market_id which is an integer like 401484315
        '''
        parameters = {
            "SearchByMarketName" : True,
            "Query" : market_name,
            "MaxResults" : 10,
            "includeOptions" : False,
            SESSION: self.auth_token,
            USERNAME: self.uname
        }

        try:
            res = requests.get(URL_SEARCH_BY_MARKET_NAME, params=parameters)
            res_data_json = res.json()
            # print(res_data_json)
        except requests.exceptions.HTTPError as e:
            raise requests.exceptions.HTTPError(e.strerror)

        if res.status_code == 200:
            return res_data_json['Markets'][0]['MarketId']

        return res_data_json['Markets'][0]['MarketId']

    def get_client_trade_history(self, maxResults):
        '''
        Makes an api call (Post request) to server to get trade history of a user
        :param maxResults: The number of latest user transaction to be shown
        :return: A dictionary : TradeHistory{'OrderId' :___, 'MarketId : ___, 'MarketName':___........} for 200 response status code
        '''
        # client_account_id, trading_acct_id, account_operator_id = self.get_user_info()
        parameters = {'Session': self.session, 'UserName': self.uname, 'TradingAccountId': self.trading_acc_id,
                  'maxResults': maxResults}
        try:
            res = requests.get(URL_LIST_CLIENT_TRADE_HISTORY, params=parameters)
            res_data_json = res.json()
            print(res_data_json)
        except requests.exceptions.HTTPError as e:
            print("Error in Get Client History")
            raise requests.exceptions.HTTPError(e.strerror)

        if 'TradeHistory' in res_data_json.keys():
            return res_data_json
        return False

if __name__ == '__main__':
    forex = Forex()
    print(forex.login('DA545354', 'Forex123', 'N.Nelson'))
    print(forex.get_margins())

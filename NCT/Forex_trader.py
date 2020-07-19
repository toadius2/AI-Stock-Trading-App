# Standard libraries
import logging
import warnings
import sys
import requests
import getpass
import json

# These imports are to removed and added to the file showing Forex GUI
import csv
import os

from enum import Enum

# Application-specific imports
import exceptions_forex as exception
import endpoints_forex as endpoints

URL_LOGIN = "https://ciapi.cityindex.com/TradingAPI/session"
URL_GET_USER_INFO = "https://ciapi.cityindex.com/TradingAPI/UserAccount/ClientAndTradingAccount"
URL_GET_CLIENT_ACCOUNT_MARGIN = "https://ciapi.cityindex.com/TradingAPI/margin/ClientAccountMargin"
URL_VALIDATE_SESSION = "https://ciapi.cityindex.com/TradingAPI/session/validate"
URL_LOGOFF = "https://ciapi.cityindex.com/TradingAPI/session/deleteSession"

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

class Forex_trader:
    # instance attributes
    def __init__(self, user="", password="", app_key=""):

        print("Object created")
        self.username = user
        self.password = password
        self.app_key = app_key
        self.session = requests.session()
        self.auth_token = None
        # self.is_twoFA_enabled = None  *2FA is Currently not supported in Forex.com
        self.headers = {
            "Content-Type": "application/json",
        }
        self.session.headers = self.headers

    def login(self):
        '''
        Logs in the User with credentials given when object is created
        :return: True and sets auth_token to a valid session ID, otherwise False
        '''
        print("Loggin in....")
        if self.username == "" or self.password == "":
            print("Unable to Login. Use correct User/Password!")
            return False

        request_body = {
            "UserName": self.username,
            "Password": self.password,
            "AppVersion": "1",
            "AppComments": "",
            "AppKey": self.app_key
        }
        # HTTP Post request to server
        try:
            res = requests.post(URL_LOGIN, json=request_body)
            res_data = res.json()
            # print("Login data: ",res_data)
        except requests.exceptions.HTTPError as e:
            print("Error while Loggin in!")
            print(e.strerror)
            raise requests.exceptions.HTTPError(e.strerror)

        # If request is successful, response will contain a key named 'Session'
        if SESSION in res_data.keys():
            self.auth_token = res_data[SESSION]
            # self.is_twoFA_enabled = res_data[IS_2FA_ENABLED]
            print("In Login - Session ID = ", self.auth_token)
            # print("IS_2FA_Enabled= ",self.is_twoFA_enabled)
            return True

        return False

    def get_user_info(self):
        '''
        Performs HTTP Get request to server to Get the user information
        :return: A dictionary having keys: isError, ClientAccountID and TradingAccountID
                If isError is set to True, ClientAccountID and TradingAccountID are not present
        '''
        if self.auth_token is None:
            print("Need to Login first!")
            return {IS_ERROR: True}

        if not self.validate_session():
            print("Session is invalid, Login again!")
            return {IS_ERROR: True}

        print('Valid session in Get user info()')
        parameters = {SESSION: self.auth_token, USERNAME: self.username}
        # session_bundle = '?Session=' + self.auth_token + '&UserName=' + self.username
        try:
            # res = requests.get(URL_GET_USER_INFO + session_bundle)
            res = requests.get(URL_GET_USER_INFO, params=parameters)
            res_data_json = res.json()
            # print(res_data_json)
        except requests.exceptions.HTTPError as e:
            print("Error in GetUserInfo!")
            print(e.strerror)
            raise requests.exceptions.HTTPError(e.strerror)

        # # If session is not valid, call login() again, then call get_user_info() again
        # if HTTP_STATUS in res_data_json.keys() and res_data_json[HTTP_STATUS] == 401 :
        #     print("Session not valid :{}".format(res_data_json[HTTP_STATUS]))
        #     print("Please Login again...")
        #     return {IS_ERROR : True}

        print("Getinfo successful  ", res_data_json[CLIENT_ACCOUNT_ID],
              res_data_json[TRADING_ACCOUNTS][0][TRADING_ACCOUNT_ID])
        # If request is successful, extract and set Account ID variables
        return {IS_ERROR: False,
                CLIENT_ACCOUNT_ID: res_data_json[CLIENT_ACCOUNT_ID],
                TRADING_ACCOUNT_ID: res_data_json[TRADING_ACCOUNTS][0][TRADING_ACCOUNT_ID],
                ACCOUNT_OPERATOR_ID: res_data_json[ACCOUNT_OPERATOR_ID]}

    def get_client_margin(self):
        '''
        Performs HTTP Get request to server
        :return: A dictionary having keys: isError, Cash, Margin
                isError is set to True in case of error response
        '''
        print("In get_client_margin()")
        if self.auth_token is None:
            print("Need to Login first!")
            return {IS_ERROR: True}

        if not self.validate_session():
            print("Session is invalid, Login again!")
            return {IS_ERROR: True}

        parameters = {SESSION: self.auth_token, USERNAME: self.username}
        try:
            res = requests.get(URL_GET_CLIENT_ACCOUNT_MARGIN, params=parameters)
            res_data_json = res.json()
        except requests.exceptions.HTTPError as e:
            print('HTTPError in GetUserInfo!')
            print(e.strerror)
            raise requests.exceptions.HTTPError(e.strerror)

        # if HTTP_STATUS in res_data_json.keys() and res_data_json[HTTP_STATUS] == 401:
        #     print("Session not valid :{}".format(res_data_json[HTTP_STATUS]))
        #     return {IS_ERROR : True}

        print('Get_client_margin successful!')
        # Retrieve Cash, Margin & CurrentcyIsoCode from the response
        return {IS_ERROR: False,
                CASH: res_data_json[CASH],
                MARGIN: res_data_json[MARGIN],
                CURRENCY: res_data_json[CURRENCY_ISO_CODE]}

    def validate_session(self):
        '''
        HTTP Post request to server to check session validation
        :return: Bool
        '''
        request_body = {
            SESSION: self.auth_token,
            USERNAME: self.username
        }
        try:
            res = requests.post(URL_VALIDATE_SESSION, json=request_body)
            res_data = res.json()
            print(res_data)

        except requests.exceptions.HTTPError:
            raise exception.LoginFailed()

        if 'IsAuthenticated' in res_data.keys() and res_data['IsAuthenticated'] is True:
            return True

        return False

    def log_off(self):
        '''
        HTTP Post request to server to Log OFF and delete session
        :return: bool
        '''
        request_body = {
            SESSION: self.auth_token,
            USERNAME: self.username
        }
        parameters = {SESSION: self.auth_token, USERNAME: self.username}
        try:
            res = requests.post(URL_LOGOFF, json=request_body, params=parameters)
            res_data = res.json()
            print(res_data)

        except requests.exceptions.HTTPError:
            raise exception.LoginFailed()

        if res_data[LOGGED_OUT]:
            return True
        else:
            return False

# Test this program for all API calls
if __name__ == '__main__':

    my_trader = Forex_trader("DA545354", "Forex123")
    if my_trader.login():
        result = my_trader.get_user_info()
        if result[IS_ERROR]:
            print("Error in get_user_info()")
        else:
            result = my_trader.get_client_margin()
            if result[IS_ERROR]:
                print("Error in get_client_margin()")
    # my_trader = Forex_trader("DA545354","Forex123","N.Nelson")
    # if my_trader.login():
    #     result = my_trader.get_user_info()

    print("End of Main")

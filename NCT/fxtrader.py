import requests

URL_LOGIN = "https://ciapi.cityindex.com/TradingAPI/session"
URL_GET_CLIENT_ACCOUNT_MARGIN = "https://ciapi.cityindex.com/TradingAPI/margin/ClientAccountMargin"

def login(username, password, appkey=""):

    request_body = {
        "UserName": username,
        "Password": password,
        "AppVersion": "1",
        "AppComments": "",
        "AppKey": appkey
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
    if "Session" in res_data.keys():
        print("Session id = ", res_data["Session"])
        return res_data["Session"]

    else:
        return False

def get_client_margin(username, sessionid):
    '''
    Performs HTTP Get request to server
    :return: A dictionary having keys: isError, Cash, Margin
            isError is set to True in case of error response
    '''

    parameters = {"Session": sessionid, "UserName": username}

    try:

        res = requests.get(URL_GET_CLIENT_ACCOUNT_MARGIN, params=parameters)
        res_data_json = res.json()
        print(res_data_json)
    except requests.exceptions.HTTPError as e:
        print(e.strerror)
        raise requests.exceptions.HTTPError(e.strerror)

    print("Status code (Getclientmargin) = ",res.status_code)
    if res.status_code == 200 :
        return res_data_json["Cash"], res_data_json["CurrencyIsoCode"]
    else:
        return False

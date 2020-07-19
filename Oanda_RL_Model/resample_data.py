import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20 as op
import pandas as pd
from collections import defaultdict
#import matplotlib
import requests


def get_oanda_account_balance(access_token):
    client = oandapyV20.API(access_token,"live")
    r = accounts.AccountList()
    #print(r)
    acct_margin = 0
    try:
        result = client.request(r)
        print(result)
        acct_id = result['accounts'][0]['id']
        print(result)
        print("account")
        print(acct_id)
        r = accounts.AccountSummary(accountID=acct_id)
        result1 = client.request(r)
        print(result1)
        acct_margin = result1.get('account', {}).get('balance')
        currency = result1.get('account', {}).get('currency')
        print(acct_margin)
        print(currency)

    except oandapyV20.exceptions.V20Error as e:
        print("error", str(e))

    return acct_margin


#get_oanda_account_balance('44674d0e89da84b65f110fd0233d1fba-e2d49ffb3046945bcde051d4213e4331')


def get_oanda_ohlcv_data(currency_pair, access_token, granularity, from_date, to_date):
    '''

    :param currency_pair: currency pair whose data is to be extracted
    :param access_token: account access token
    :param granularity: time interval for which data is needed
    :param from_date: start time
    :param to_date: end time
    :return: ohlcv data as a dataframe
    '''

    ohlcv_data = 0
    parameters = {
        "granularity": granularity,
        "price": "M",
        "from": from_date,
        "to": to_date
    }
    try:
        client = op.API(access_token,"live")
        req = instruments.InstrumentsCandles(currency_pair, parameters)
        client.request(req)
        data_dict = req.response
        ''' extract data with candles'''
        data = data_dict['candles']

        ''' convert dict to data frame'''
        raw_data_dict = pd.DataFrame.from_dict(data)
        ''' copy columns volume, time and mid into another data frame, mid is in the form of dict'''
        data_vol_time_mid_prices = raw_data_dict[['volume','time','mid']].copy()

        ''' extracting mid column and copying into another data frame'''
        data_mid = data_vol_time_mid_prices['mid'].apply(pd.Series)

        ''' droping mid dicts from the first data frame'''
        new = data_vol_time_mid_prices.drop(['mid'], axis=1)

        ''' concatinating the two data frames'''
        ohlcv_data = pd.concat([data_mid, new], axis=1, sort=False)

        ''' renaming columns'''
        ohlcv_data = ohlcv_data.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'time': 'timestamp'})

        ''' rearranging columns'''
        ohlcv_data = ohlcv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print(ohlcv_data)
    except oandapyV20.exceptions.V20Error as e:
        print("error", str(e))

    return ohlcv_data



price_his = get_oanda_ohlcv_data("EUR_USD", '44674d0e89da84b65f110fd0233d1fba-e2d49ffb3046945bcde051d4213e4331', 'M1', 1567104070, 1567129270)

def label_perfect_trades(price_history, trigger_price, new_training=False):
    """
    Labels perfect trades as Peaks(2), Valleys(1) or Neutral(0)
    :param price_history: Dataframe of OHLC
    :param trigger_price:
    :param new_training:
    :return:
    """
    label = []
    dict1 = {}
    dict2 = {}
    dict_count = defaultdict(int)
    data = price_history['close'].values
    trigger = data[0]
    for i in range(0, len(data)):
        # print("Iteration", i)
        # print("count", dict_count[2], dict_count[1])
        if float(data[i]) > (1.005 * float(trigger)):
            if dict_count[1] >= 2:
                dict_count[1] = 1
                dict2 = {}
            trigger = data[i]
            label.append(2)
            if 2 in dict2:
                label[dict2[2]] = 0
                dict2[2] = i
            else:
                dict2[2] = i
            if 2 not in dict_count:
                dict_count[2] = 1
            else:
                dict_count[2] += 1
        elif float(data[i]) < 0.995 * float(trigger):
            if dict_count[2] >= 2:
                dict_count[2] = 1
                dict1 = {}
            trigger = data[i]
            label.append(1)
            if 1 in dict1:
                label[dict1[1]] = 0
                dict1[1] = i
            else:
                dict1[1] = i
            if 1 not in dict_count:
                dict_count[1] = 1
            else:
                dict_count[1] += 1
        else:
            # trigger = data[i]
            label.append(0)
    label_df = pd.DataFrame(label, columns=['target'])
    priceHistory = pd.concat([price_history, label_df], 1)
    return priceHistory.fillna(0), trigger_price


new_price_history, triggerprice = label_perfect_trades(price_his, 0.000)
new_price_history.to_csv('test.csv', sep='\t', encoding='utf-8')
print(new_price_history)

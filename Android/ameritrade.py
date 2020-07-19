import requests
import json
import urllib
import time
import webbrowser as br
from splinter import Browser
from flask import Flask, request


class Ameritrade:
    def __init__(self):
        self.session = requests.session()
        self.session.headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

    # def get_auth_code(self):
    #     payload = {
    #         'response_type': 'code',
    #         'redirect_uri': 'https://localhost:8443',
    #         'client_id': 'A84UCPOEY9BKT4VM3MYASTSUZJATUSSM' + '@AMER.OAUTHAP'
    #     }
    #
    #     url = 'https://auth.tdameritrade.com/auth?'
    #     p = requests.Request('GET', url, params=payload).prepare()
    #     print(p.url)
    #     br.open(p.url)
    #     app = Flask(__name__)
    #     app.run(debug=True, port=8443)
    #     code = request.args.get('code')

    def get_auth_code(self, id, password):
        path = {
            'executable_path': 'geckodriver'
        }
        br = Browser('firefox', **path)
        payload = {
            'response_type': 'code',
            'redirect_uri': 'https://localhost:8443',
            'client_id': 'A84UCPOEY9BKT4VM3MYASTSUZJATUSSM' + '@AMER.OAUTHAP'
        }

        url = 'https://auth.tdameritrade.com/auth?'
        p = requests.Request('GET', url, params=payload).prepare()
        endpoint = p.url

        br.visit(endpoint)
        username = br.find_by_id('username').first.fill(id)
        password = br.find_by_id('password').first.fill(password)
        submit = br.find_by_id('accept').first.click()

        time.sleep(1)
        while 'code=' not in br.url:
            pass

        new_url = br.url
        parse_url = urllib.parse.unquote(new_url.split('code=')[1])
        return parse_url

    def auth(self, id, password):
        url = 'https://api.tdameritrade.com/v1/oauth2/token'
        code = self.get_auth_code(id, password)
        print(code)

        params = {
            'grant_type': 'authorization_code',
            'access_type': 'offline',
            'code': code,
            'client_id': 'A84UCPOEY9BKT4VM3MYASTSUZJATUSSM',
            'redirect_uri': 'https://localhost:8443'
        }
        res = self.session.post(url, params=params)
        return res.json()

if __name__ == '__main__':
    am = Ameritrade()
    print(am.auth('Noahnelson13', 'CompeteToWin*13'))

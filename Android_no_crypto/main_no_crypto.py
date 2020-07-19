from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.config import Config
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty
from kivy.uix.button import Button
from kivy.core.window import Window
import requests
import re
import json
import os
import threading
import webbrowser
import re
#APIs
from forex import Forex
from crypto import Crypto
from Robinhood import Robinhood
from particle import Particle
from particle_publish import Particle_publish
from payment import Square

# cd documents\kivy_app
# kivy_venv\scripts\activate

fx = Forex()
robin = Robinhood()
crypto = Crypto()
square = Square()

class MainWindow(Screen):
    def start_listening(self):
        credentials = None
        if not os.path.exists('payment.txt'):
            self.parent.current = 'paymentOptions'
            return

        if not self.check_paid_status():
            if self.check_order_id() != 'COMPLETED':
                grid = GridLayout(cols=1)
                button1 = Button(text='Pay now', size_hint=(1, .1))
                button2 = Button(text='OK', size_hint=(1, .1))
                grid.add_widget(button1)
                grid.add_widget(button2)
                popup = Popup(title='Payment Missing', content=grid, size_hint=(.5,.5))
                button1.bind(on_press=self.change_screen)
                button1.bind(on_press=popup.dismiss)
                button2.bind(on_press=popup.dismiss)
                popup.open()
            else:
                credentials = self.check_file_conts()
        else:
            credentials = self.check_file_conts()
            print(credentials)

        if credentials is not None:
            log_result = robin.login(credentials['Runame'], credentials['Rpassw'])
            if log_result == 'mfa required':
                self.parent.current = 'tfa_robin'
            elif not log_result:
                content = Button(text='OK', size_hint= (1, .1))
                popup = Popup(title='Check Robinhood credentials', content=content, size_hint=(.5, .5))
                content.bind(on_press=popup.dismiss)
                popup.open()
                return

            particle = Particle(credentials)
            auth = particle.auth()
            if auth == "success":
                content = Button(text='OK', size_hint= (1, .1))
                popup = Popup(title='Listening started', content=content, size_hint=(.5, .5))
                content.bind(on_press=popup.dismiss)
                popup.open()
                t1 = threading.Thread(target=particle.get_events_by_name, args=('String_msg_from_RL',), daemon=True)
                if self.t1.isAlive():
                    particle.thread_run = False
                else:
                    particle.thread_run = True
                    self.t1.start()
            else:
                content = Button(text='OK', size_hint=(1, .1))
                popup = Popup(title='Something went wrong. Try again later', content=content, size_hint=(.5, .5))
                content.bind(on_press=popup.dismiss)
                popup.open()

    def change_screen(self, *args):
        self.parent.current = 'paymentOptions'

    def check_order_id(self):
        print('Getting order id')
        order_id = ''
        txn_id = ''
        with open('payment.txt', 'r') as json_file:
            data = json.load(json_file)
            try:
                order_id = data['order_id']
            except Exception as e:
                print(str(e))

        if order_id == '':
            return
        #Check square id first
        order = square.retrieve_order(order_id)
        try:
            if order[0]['code'] == 'ORDER_EXPIRED':
                return "not paid"
        except Exception as e:
            print(str(e))

        if len(order['orders']) < 1:
            return "not paid"
        else:
            data['paid'] = True
            with open('payment.txt', 'w') as f:
                json.dump(data, f)
            return order['orders'][0]['state']


    def check_paid_status(self):
        with open('payment.txt') as file:
            data = json.load(file)
            if 'paid' in data.keys():
                return data['paid']
            else:
                return False

    def check_file_conts(self):
        credentials = {
            'crypto': [],

            'Funame': '',
            'Fpassw': '',
            'Fappkey': '',

            'Runame': '',
            'Rpassw': ''
        }
        data = {}
        crypto_data = []

        # check crypto creds
        if os.path.exists('crypto.txt'):
            json_file = open('crypto.txt', 'r')
            try:
                crypto_data = json.load(json_file)
            except Exception as e:
                print(e)
            json_file.close()
            if len(crypto_data) > 0:
                # Crypto complete
                credentials['crypto'] = crypto_data
            else:
                self.parent.current = 'credsCrypto'
                return None
        else:
            self.parent.current = 'credsCrypto'
            return None

        # check forex creds
        if os.path.exists('forex.txt'):
            json_file = open('forex.txt', 'r')
            try:
                data = json.load(json_file)
            except Exception as e:
                print(e)
            json_file.close()
            if 'Funame' in data.keys() and 'Fpassw' in data.keys() and 'Fappkey' in data.keys():
                # forex complete
                credentials['Funame'] = data['Funame']
                credentials['Fpassw'] = data['Fpassw']
                credentials['Fappkey'] = data['Fappkey']
            else:
                self.parent.current = 'credsForex'
                return None
        else:
            self.parent.current = 'credsForex'
            return None

        # check Robin creds
        if os.path.exists('robin.txt'):
            json_file = open('robin.txt', 'r')
            try:
                data = json.load(json_file)
            except Exception as e:
                print(e)
            json_file.close()
            if 'Runame' in data.keys() and 'Rpassw' in data.keys():
                # Robin complete
                credentials['Runame'] = data['Runame']
                credentials['Rpassw'] = data['Rpassw']
            else:
                self.parent.current = 'credsRobin'
                return None
        else:
            self.parent.current = 'credsRobin'
            return None

        return credentials
        #
        # grid = GridLayout(cols=1)
        # button = Button(text="OK", size_hint=(1, .1))
        # grid.add_widget(Label(text=apiKey + '\n' + secretKey, size_hint=(1, .1)))
        # grid.add_widget(button)
        # popup = Popup(title="Check credentials", content=grid, size_hint=(.5, .5))
        # button.bind(on_press=popup.dismiss)
        # popup.open()

class TermsofService(Screen):
    def __init__(self, **kwargs):
        super(TermsofService, self).__init__(**kwargs)

    def agreed(self):
        if os.path.exists('terms.txt'):
            with open('terms.txt', 'w') as f:
                data = {'agreed': True}
                json.dump(data, f)
            if variables.free:
                content = Button(text='OK', size_hint=(1, .1))
                popup = Popup(title='100% discount applied. Feel free to use our service!', content=content, size_hint=(.5, .5))
                content.bind(on_press=popup.dismiss)
                content.bind(on_press=self.change_screen)
                popup.open()
        else:
            return
        self.manager.current = 'paymentOptions'

    def change_screen(self, *args):
        self.manager.current = 'main'

class PaymentOptions(Screen):
    def __init__(self, **kwargs):
        super(PaymentOptions, self).__init__(**kwargs)
        self.apply_code = False

    def check(self):
        # check whether payments have gone through or not
        if os.path.exists('payment.txt'):
            message = self.check_order_id()
            self.ids['paymentStatus'].text = message
        else:
            self.ids['paymentStatus'].text = 'Not Paid'

    def check_discount(self):
        # Check if discount code is valid
        discount_codes = ['rcvi13', 'aiv13', 'v13er', 'v13ai', 'aivision', 'vision3ai', 'visionai', 'ai2019', 'dsho', 'vthirteenai',
            'V13Ai', 'Thirteenvision', 'Alphavision13', '2019ai', '5ai', 'AIVision13', 'xavai', 'ngthon13', 'songAI', 'prust13ai', 'kyle13', 'wolfAI',
            'CottAI13', 'MinAI', 'Sami13', 'PradVision13']
        code = self.ids['discountCode'].text
        if code in discount_codes:
            self.ids['discountCodeValidity'].text = 'Valid'
            self.apply_code = True
        elif code == 'NCT4FREE':
            self.ids['discountCodeValidity'].text = '100% applied'
            self.free = True
            variables.free = True
            self.pay()
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please make sure to check terms of service', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()
        else:
            self.ids['discountCodeValidity'].text = 'Invalid'

    def pay(self):
        # Square payment
        if os.path.exists('terms.txt'):
            f = open('terms.txt', 'r')
            try:
                agree = json.load(f)['agreed']
            except Exception:
                agree = False
            f.close()
        else:
            return
        if not agree:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='You must agree to terms of service', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()
            return
        if self.free:
            self.change_paid_status()
        else:
            order_id, url = square.create_url(apply_discount=self.apply_code)
            self.save_order_id(order_id)
            self.parent.current = 'main'
            webbrowser.open(url)

    def change_paid_status(self):
        data = None
        to_write = {"paid": True}
        if not os.path.exists('payment.txt'):
            with open('payment.txt', 'w') as f:
                json.dump(to_write, f)
                print('dumped - initial')
        else:
            with open('payment.txt', 'r+') as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(e)
                if data is None:
                    json.dump(to_write, f)
                    print('dumped')
                else:
                    data['paid'] = True
                    json.dump(data, f)

    def save_order_id(self, order_id):
        data = {}
        data['sq_id'] = order_id
        data['paid'] = False
        f = open('payment.txt', 'w')
        json.dump(data, f)
        f.close()

    def check_order_id(self):
        order_id = ''
        with open('payment.txt', 'r') as json_file:
            data = json.load(json_file)
            try:
                order_id = data['order_id']
            except Exception as e:
                print(str(e))

        if order_id == '':
            return 'Not Paid'
        #Check square id first
        order = square.retrieve_order(order_id)
        try:
            if order[0]['code'] == 'ORDER_EXPIRED':
                return "Not Paid"
        except Exception as e:
            print(str(e))

        if len(order['orders']) < 1:
            return "Not Paid"
        else:
            data['paid'] = True
            with open('payment.txt', 'w') as f:
                json.dump(data, f)
            return 'Payment Accepted'

class CredentialCheckCrypto(Screen):
    def __init__(self, **kwargs):
        super(CredentialCheckCrypto, self).__init__(**kwargs)
        self.exchange = ''
        self.exchanges = []
        self.currently_available = ['binance', 'binanceus']
        self.part = Particle_publish()
        self.email = ''

    def write_file(self):
        data = {}
        data['apiKey'] = self.ids['CrChAPIKey'].text
        data['secretKey'] = self.ids['CrChSecretKey'].text
        data['exchange'] = self.exchange
        if data['exchange'] not in self.currently_available:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Exchange currently unavailable. Support will arrive in 24-48 hours', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            self.part.auth()
            print(self.part.publish_event(event_name='Exchange_request', data=self.exchange))
            popup.open()
        if data not in self.exchanges:
            self.exchanges.append(data)
        with open('crypto.txt', 'w') as json_file:
            json.dump(self.exchanges, json_file)
        self.manager.current = "main"

    def add(self):
        data = {}
        data['apiKey'] = self.ids['CrChAPIKey'].text
        data['secretKey'] = self.ids['CrChSecretKey'].text
        data['exchange'] = self.exchange
        if data['exchange'] not in self.currently_available:
            button = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Exchange currently unavailable. Support will arrive in 24-48 hours', content=button, size_hint=(.5, .5))
            button.bind(on_press=popup.dismiss)
            self.part.auth()
            print(self.part.publish_event(event_name='Exchange_request', data=self.exchange))
            popup.open()
        self.exchanges.append(data)

    def handle_crypto_choice(self):
        self.email = self.ids['email'].text
        if validate_email(self.email):
            self.part.auth()
            print(self.part.publish_event(event_name='Client_email', data=self.email))    
        self.write_file()

    def validate_email(self, email):
        regex = '^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
        if(re.search(regex, email)):
            return True
        else:
            return False

    def on_spinner_select_exchange(self, spinner_choice):
        self.exchange = spinner_choice

class CredentialCheckForex(Screen):
    def write_file(self):
        data = {}
        data['Funame'] = self.ids['CrChUname'].text
        data['Fpassw'] = self.ids['CrChPassw'].text
        data['Fappkey'] = self.ids['CrChAppkey'].text
        with open('forex.txt', 'w') as json_file:
            json.dump(data, json_file)
        self.manager.current = "main"

class CredentialCheckRobin(Screen):
    def write_file(self):
        data = {}
        data['Runame'] = self.ids['CrChRobinName'].text
        data['Rpassw'] = self.ids['CrChRobinPassw'].text
        with open('robin.txt', 'w') as json_file:
            json.dump(data, json_file)
        self.manager.current = "main"


class DonationWindow(Screen):
    def __init__(self, **kwargs):
        super(DonationWindow, self).__init__(**kwargs)
        self.country = ''
        self.cause = ''
        self.percentage = ''

    def donation_details(self):
        country = self.ids['country'].text;
        cause = self.ids['cause'].text;
        percentage = self.ids['percentage'].text;

    def on_spinner_select_country(self, chooser):
        self.country = chooser
    def on_spinner_select_cause(self, chooser):
        self.cause = chooser
    def on_spinner_select_percentage(self, chooser):
        self.percentage = chooser

class CurrLoginWindow(Screen):
    def __init__(self, **kwargs):
        super(CurrLoginWindow, self).__init__(**kwargs)
        self.exchange = ''
        self.stock_exchange = ''
        self.fx = fx
        self.robin = robin
        self.crypto = crypto

    ## Forex API

    def loginForex(self):
        funame = self.ids['funame'].text
        fpassw = self.ids['fpassw'].text
        appkey = self.ids['appkey'].text
        if self.fx.login(funame, fpassw, appkey):
            self.manager.current = 'forexPort'
        else:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check credentials and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

    ## Crypto API

    def on_spinner_select(self, exch):
        self.exchange = exch

    def loginCrypto(self):
        # apikey = "ihe8Ac7EWc1Gvme4Pkub00Uml1qdVCYAF2efVYjB3O1RBwm778lzPGYbJm8g3LgL"
        # secretkey = "pGPmiVVLlA6CqhmHyLvS0IGC7roTpUugjnBX2dmQ2rdLjiVdepB2OIiezM9fVkx1"
        apikey = self.ids["CAPIKey"].text
        skey = self.ids["CSecretKey"].text
        if self.exchange == '' or apikey =='' or skey == '':
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please enter all details and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()
        else:
            result = self.crypto.get_account_balance(self.crypto.gen_account_id(apikey, skey, self.exchange))
            if result == 'error':
                content = Button(text='OK', size_hint=(1, .1))
                popup = Popup(title='Please check details and try again', content=content, size_hint=(.5, .5))
                content.bind(on_press=popup.dismiss)
                popup.open()
            else:
                self.manager.current = 'cryptoPort'


    ## Robinhood API

    def on_spinner_select_exchange(self, selection):
        self.stock_exchange = selection

    def handle_login_stocks(self):
        if self.stock_exchange == 'Robinhood':
            self.loginRobin()
        else:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='More exchanges coming soon!', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

    def loginRobin(self):
        runame = self.ids["runame"].text
        rpassw = self.ids["rpassw"].text
        log_result = self.robin.login(username=runame, password=rpassw)
        print(log_result)
        if log_result == True:
            self.manager.current = 'robinPort'
        elif log_result == False:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check credentials and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()
        elif log_result == 'mfa required':
            self.manager.current = 'tfa_robin'

class RobinMFAScreen(Screen):
    # nn131983
    # Hittingstride*13
    def tfa_login(self):
        sms_code = self.ids['sms_code'].text
        login_final = robin.mfa_login(sms_code)
        if login_final == True:
            self.manager.current = 'robinPort'
        else:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check credentials and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

class CryptoPortfolio(Screen):
    def __init__(self, **kwargs):
        super(CryptoPortfolio, self).__init__(**kwargs)
        self.acc = None
        self.ex = ''

    def change_crypto(self):
        exhange = self.ids['ExName'].text.lower()
        if exchange == '' or exchange not in ccxt.exchanges:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check exchange name and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()
        self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, exchange)
        self.ex = exchange

    def account_balance(self):
        self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
        bal = crypto.get_account_balance(self.acc)
        if bal == 'error':
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check exchange and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()
        grid = GridLayout(cols=1, rows=len(bal.keys()))
        dismiss_but = Button(text="Dismiss", size_hint=(1, .1))
        for item in bal:
            grid.add_widget(Label(text=item + ': ' + str(bal[item])))
        grid.add_widget(dismiss_but)
        popup = Popup(title='Account Information', content=grid, size_hint=(.5, .5))
        dismiss_but.bind(on_press=popup.dismiss)
        popup.open()

    def sell_market(self):
        if self.acc is None:
            self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
        amount = int(self.ids['AmountMarket'].text)
        market_id = self.ids['MarketIdMarket'].text
        x = crypto.sell_order_market(self.acc, market_id, amount)
        if x == 'error':
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

    def buy_market(self):
        if self.acc is None:
            self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
        amount = int(self.ids['AmountMarket'].text)
        market_id = self.ids['MarketIdMarket'].text
        x = crypto.buy_order_market(self.acc, market_id, amount)
        if x == 'error':
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

    def sell_limit(self):
        if self.acc is None:
            self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
        amount = int(self.ids['AmountLimit'].text)
        price = int(self.ids['PriceLimit'].text)
        market_id = self.ids['MarketIdLimit'].text
        x = crypto.sell_order_limit(self.acc, market_id, amount, price)
        if x == 'error':
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

    def buy_limit(self):
        if self.acc is None:
            self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
        amount = int(self.ids['AmountLimit'].text)
        price = int(self.ids['PriceLimit'].text)
        market_id = self.ids['MarketIdLimit'].text
        x = crypto.buy_order_limit(self.acc, market_id, amount, price)
        if x == 'error':
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()



class ForexPortfolio(Screen):
    #DA545354
    def __init__(self, **kwargs):
        super(ForexPortfolio, self).__init__(**kwargs)
        self.quantity = 0
        self.exchange = ''
        self.order_id = ''

    def logoutForex(self):
        self.validateUserFor()
        print(fx.logoff())

    def user_info_Forex(self):
        uinfo = fx.user_info()
        grid = GridLayout(cols=1, rows=4)
        dismiss_but = Button(text="Dismiss", size_hint=(1, .1))
        for item in uinfo:
            grid.add_widget(Label(text=item + ': ' + str(uinfo[item])))
        grid.add_widget(dismiss_but)
        popup = Popup(title='Account Information', content=grid, size_hint=(.5, .5))
        dismiss_but.bind(on_press=popup.dismiss)
        popup.open()

    def get_margins_Forex(self):
        margin = fx.get_margins()
        grid = GridLayout(cols=1, rows=4)
        dismiss_but = Button(text="Dismiss", size_hint=(1, .1))
        for item in margin:
            grid.add_widget(Label(text=item + ': ' + str(margin[item])))
        grid.add_widget(dismiss_but)
        popup = Popup(title='Margins', content=grid, size_hint=(.5, .5))
        dismiss_but.bind(on_press=popup.dismiss)
        popup.open()

    def validateUserFor(self):
        print(fx.validate_session())


class ForexBuyWindow(Popup):

    def buy_order_Forex(self):
        market_name = self.ids['marketname'].text
        qnty = self.ids['quantity'].text
        market_id = fx.get_marketId_by_name(market_name)
        print(market_id)  # 401484347 for EUR/USD
        order_id = fx.buy_order(market_id, qnty)

        if order_id == 'error':
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check all details and retry', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

        elif order_id == 'unsuccessful':
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Order unsuccessful. Try again later.', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

        else:
            dismiss_but = Button(text="Ok", size_hint=(1, .1))
            grid = GridLayout(rows=2)
            grid.add_widget(Label(text=str(order_id)))
            grid.add_widget(dismiss_but)
            popup = Popup(title='Order successful!', content=grid, size_hint=(.5, .5))
            dismiss_but.bind(on_press=popup.dismiss)
            popup.open()


class ForexSellWindow(Popup):

    def sell_order_Forex(self):

        market_name = self.ids['marketname'].text
        qnty = self.ids['quantity'].text
        market_id = fx.get_marketId_by_name(market_name)
        print(market_id)  # 401484347 for EUR/USD
        order_id = fx.sell_order(market_id, 0, qnty)

        if order_id == 'error':
            dismiss_but = Button(text="Ok", size_hint=(1, .1))
            grid = GridLayout(rows=2)
            grid.add_widget(dismiss_but)
            popup = Popup(title='Please check all details and retry', content=grid, size_hint=(.5, .5))
            dismiss_but.bind(on_press=popup.dismiss)
            popup.open()

        elif order_id == 'unsuccessful':
            dismiss_but = Button(text="Ok", size_hint=(1, .1))
            grid = GridLayout(rows=2)
            grid.add_widget(dismiss_but)
            popup = Popup(title='Order unsuccessful. Try again later.', content=grid, size_hint=(.5, .5))
            dismiss_but.bind(on_press=popup.dismiss)
            popup.open()

        else:
            dismiss_but = Button(text="Ok", size_hint=(1, .1))
            grid = GridLayout(rows=2)
            grid.add_widget(Label(text=str(order_id)))
            grid.add_widget(dismiss_but)
            popup = Popup(title='Order successful!', content=grid, size_hint=(.5, .5))
            dismiss_but.bind(on_press=popup.dismiss)
            popup.open()


class RobinhoodPortfolio(Screen):
    def logoutRobin(self):
        print(robin.logout())

    def get_investment_profile(self):
        print(robin.investment_profile())

    def get_stock_quotes(self):
        stocks = self.ids["stockname"].text
        stocks = stocks.replace(',', '')
        print(robin.quotes_data(stocks.split()))

    def get_portfolio(self):
        print(robin.portfolios())

    def buy_order(self, ticker, amount):
        try:
            stock_instrument = robin.instruments(ticker)[0]
            buy_order = robin.place_buy_order(stock_instrument, amount)
        except Exception as e:
            print(str(e))
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

        return buy_order

    def sell_order(self, ticker, amount):
        try:
            stock_instrument = robin.instruments(ticker)[0]
            sell_order = robin.place_sell_order(stock_instrument, amount)
        except Exception as e:
            print(str(e))
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

        return sell_order



class WindowManager(ScreenManager):
    pass

class NCTTabbedPanel2(TabbedPanel):
    pass

class NCTTabbedPanel(TabbedPanel):
    pass

class NCTTabbedPanel1(TabbedPanel):
    pass

# class TableHeader(Label):
#     pass
#
# class CoinRecord(Label):
#     pass

class CryptoGrid(GridLayout):
    def __init__(self, **kwargs):
        super(CryptoGrid, self).__init__(**kwargs)
        self.data = ''
        self.fetch_data_from_marketCap()
        self.display_current_data()

    def fetch_data_from_marketCap(self):
        try:
            api_request = requests.get("https://api.coinmarketcap.com/v1/ticker/")
            api = json.loads(api_request.content)
            self.data = api
        except Exception as e:
            print(str(e))
            self.data = [
                {'Name': 'Bitcoin', 'Rank': '1', "Current Price": '111', "Price Paid": '99', 'Profit/Loss': '23',
                 '1 HR change': '43', '24 HR change': '100', '7day change': '1000', 'Current value': '1567'},
                {'Name': 'Ripple', 'Rank': '2', "Current Price": '111', "Price Paid": '99', 'Profit/Loss': '23',
                 '1 HR change': '43',
                 '24 HR change': '100', '7day change': '1000', 'Current value': '1567'},
                {'Name': 'Litcoin', 'Rank': '3', "Current Price": '111', "Price Paid": '99', 'Profit/Loss': '23',
                 '1 HR change': '43',
                 '24 HR change': '100', '7day change': '1000', 'Current value': '1567'},
                {'Name': 'EOS', 'Rank': '4', "Current Price": '111', "Price Paid": '99', 'Profit/Loss': '23',
                 '1 HR change': '43',
                 '24 HR change': '100', '7day change': '1000', 'Current value': '1567'},
            ]

    def display_current_data(self):
        pass

    # def create_header_current(self):
    #     return ['Name', 'Rank', 'Price', 'Change1hr', 'Change24hr', 'Change7d']
    #
    # def create_coin_current(self, i):
    #     first_column = self.data[i]['symbol']
    #     second_column = self.data[i]['rank']
    #     third_column = self.data[i]['price_usd']
    #     fourth_column = self.data[i]['percent_change_1h']
    #     fifth_column = self.data[i]['percent_change_24h']
    #     sixth_column = self.data[i]['percent_change_7d']
    #     return [first_column, second_column, third_column, fourth_column, fifth_column, sixth_column]


class MyGrid(GridLayout):
    def __init__(self, **kwargs):
        super(MyGrid, self).__init__(**kwargs)
        self.fetch_data_from_database()
        self.display_data()

    def fetch_data_from_database(self):
        # using dummy data
        self.data = [
            {'Name': 'Bitcoin', 'Rank': '1', "Current Price": '111', "Price Paid": '99', 'Profit/Loss': '23',
             '1 HR change': '43', '24 HR change': '100', '7day change': '1000', 'Current value': '1567'},
            {'Name': 'Ripple', 'Rank': '2', "Current Price": '111', "Price Paid": '99', 'Profit/Loss': '23',
             '1 HR change': '43',
             '24 HR change': '100', '7day change': '1000', 'Current value': '1567'},
            {'Name': 'Litcoin', 'Rank': '3', "Current Price": '111', "Price Paid": '99', 'Profit/Loss': '23',
             '1 HR change': '43',
             '24 HR change': '100', '7day change': '1000', 'Current value': '1567'},
            {'Name': 'EOS', 'Rank': '4', "Current Price": '111', "Price Paid": '99', 'Profit/Loss': '23',
             '1 HR change': '43',
             '24 HR change': '100', '7day change': '1000', 'Current value': '1567'},
        ]

    def display_data(self):
        for i in range(len(self.data)):
            if i < 1:
                row = self.create_header(i)
            else:
                row = self.create_coin_info(i)
            for item in row:
                wid = Label(text=item, size_hint=(1/9, .1))
                self.add_widget(wid)

    def create_header(self, i):
        return ['Name', 'Rank', 'Current Price', 'Price Paid', 'Profit/Loss', '1 HR change', '24 HR change',
                '7day change', 'Current value']

    def create_coin_info(self, i):
        first_column = self.data[i]['Name']
        second_column = self.data[i]['Rank']
        third_column = self.data[i]['Current Price']
        fourth_column = self.data[i]['Price Paid']
        fifth_column = self.data[i]['Profit/Loss']
        sixth_column = self.data[i]['1 HR change']
        seventh_column = self.data[i]['24 HR change']
        eighth_column = self.data[i]['7day change']
        nineth_column = self.data[i]['Current value']
        return [first_column, second_column, third_column, fourth_column, fifth_column, sixth_column, seventh_column,
                eighth_column, nineth_column]




kv = Builder.load_file("gui_no_crypto.kv")


class MyMainApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    MyMainApp().run()

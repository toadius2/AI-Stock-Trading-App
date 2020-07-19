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
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
import requests
import re
import json
import os
import threading
import webbrowser
import variables
#APIs
from forex import Forex
from crypto import Crypto
from Robinhood import Robinhood
from particle import Particle
from particle_publish import Particle_publish
from payment import Square
from crypto_payment import CryptoPayment
from binance import Binance
from Alpaca import Alpaca
from Exante import Exante
from Kiteconnect import Kiteconnect

# cd documents\kivy_app
# kivy_venv\scripts\activate

# Sign in for crypto portfolio - overlay

fx = Forex()
robin = Robinhood()
crypto = Crypto()
square = Square()
crypto_payment = CryptoPayment()
alpaca = Alpaca() # need to pass initialization parameters
exante = Exante() # need to pass initialization parameters
kiteconnect = Kiteconnect() # need to pass initialization parameters

class EmailBlock(Screen):
    def __init__(self, **kwargs):
        super(EmailBlock, self).__init__(**kwargs)
        if os.path.exists('email.txt'):
            f = open('email.txt', 'r+')
            data = json.load(f)
            if data['email_entered']:
                Clock.schedule_once(self.change_screen, 1)

    def change_screen(self, dt):
        self.manager.current = 'main'

    def handle_email(self):
        email = self.ids['emailInitial'].text
        if self.validate_email(email):
            with open('email.txt', 'w') as f:
                data = {'email_entered': True}
                json.dump(data, f)
            part = Particle_publish()
            part.auth()
            part.publish_event('Client_email', data=email)
            self.manager.current = 'main'
        else:
            self.ids['emailValidity'].text = 'Invalid email. Please try again'

    def validate_email(self, email):
        regex = '^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
        if(re.search(regex, email)):
            return True
        else:
            return False

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
            for ex in credentials['stock']:
                if ex['exchange'] == 'Robinhood':
                    robin_name = ex['username']
                    robin_pass = ex['password']
                    log_result = robin.login(robin_name, robin_pass)
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
                if t1.isAlive():
                    particle.thread_run = False
                    content = Button(text='OK', size_hint= (1, .1))
                    popup = Popup(title='Listening stopped', content=content, size_hint=(.5, .5))
                    content.bind(on_press=popup.dismiss)
                    popup.open()
                else:
                    particle.thread_run = True
                    t1.start()
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
                txn_id = data['txn_id']
            except Exception as e:
                print(str(e))

        if txn_id == '' and order_id == '':
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

        #Check crypto_payment id
        try:
            order_status = crypto_payment.get_tx_info(txn_id)
        except Exception as e:
            print(str(e))
        if order_status['result']['status'] == 1:
            data['paid'] = True
            with open('payment.txt', 'w') as f:
                json.dump(data, f)
            return
        else:
            return 'not paid'

    def check_paid_status(self):
        if os.path.exists('payment.txt'):
            with open('payment.txt', 'r') as file:
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

            'stock': []
        }
        # check stored information
        if os.path.exists('crypto.txt'):
            with open('crypto.txt', 'r') as f:
                crypto_data = json.load(f)
            credentials['crypto'] = crypto_data

        if os.path.exists('forex.txt'):
            with open('forex.txt', 'r') as f:
                forex_data = json.load(f)
            credentials['Funame'] = forex_data['Funame']
            credentials['Fpassw'] = forex_data['Fpassw']
            credentials['Fappkey'] = forex_data['Fappkey']

        if os.path.exists('stock.txt'):
            with open('stock.txt', 'r') as f:
                stock_data = json.load(f)
            credentials['stock'] = stock_data

        # If nothing found, then prompt entry
        # If at least one entry, then continue
        if len(credentials['crypto']) < 1 and len(credentials['Funame']) < 1 and len(credentials['stock']) < 1:
            # all empty
            self.parent.current = 'credsCrypto'
            return None
        else:
            # at least 1 filled in
            return credentials



    # # def check_file_conts(self):
    #     credentials = {
    #         'crypto': [],
    #
    #         'Funame': '',
    #         'Fpassw': '',
    #         'Fappkey': '',
    #
    #         'Runame': '',
    #         'Rpassw': ''
    #     }
    #     data = {}
    #     crypto_data = []
    #
    #     # check crypto creds
    #     if os.path.exists('crypto.txt'):
    #         json_file = open('crypto.txt', 'r')
    #         try:
    #             crypto_data = json.load(json_file)
    #         except Exception as e:
    #             print(e)
    #         json_file.close()
    #         if len(crypto_data) > 0:
    #             # Crypto complete
    #             credentials['crypto'] = crypto_data
    #         else:
    #             self.parent.current = 'credsCrypto'
    #             return None
    #     else:
    #         self.parent.current = 'credsCrypto'
    #         return None
    #
    #     # check forex creds
    #     if os.path.exists('forex.txt'):
    #         json_file = open('forex.txt', 'r')
    #         try:
    #             data = json.load(json_file)
    #         except Exception as e:
    #             print(e)
    #         json_file.close()
    #         if 'Funame' in data.keys() and 'Fpassw' in data.keys() and 'Fappkey' in data.keys():
    #             # forex complete
    #             credentials['Funame'] = data['Funame']
    #             credentials['Fpassw'] = data['Fpassw']
    #             credentials['Fappkey'] = data['Fappkey']
    #         else:
    #             self.parent.current = 'credsForex'
    #             return None
    #     else:
    #         self.parent.current = 'credsForex'
    #         return None
    #
    #     # check Robin creds
    #     if os.path.exists('stock.txt'):
    #         json_file = open('stock.txt', 'r')
    #         try:
    #             data = json.load(json_file)
    #         except Exception as e:
    #             print(e)
    #         json_file.close()
    #         if 'Runame' in data.keys() and 'Rpassw' in data.keys():
    #             # Robin complete
    #             credentials['Runame'] = data['Runame']
    #             credentials['Rpassw'] = data['Rpassw']
    #         else:
    #             self.parent.current = 'credsRobin'
    #             return None
    #     else:
    #         self.parent.current = 'credsRobin'
    #         return None
    #
    #     return credentials
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
        self.free = False

    def check(self):
        # check whether payments have gone through or not
        if os.path.exists('payment.txt'):
            message = self.check_order_id()
            self.ids['paymentStatus'].text = str(message)
        else:
            self.ids['paymentStatus'].text = 'Not Paid'

    def check_discount(self):
        # Check if discount code is valid
        discount_codes = ['rcvi13', 'aiv13', 'v13er', 'v13ai', 'aivision', 'vision3ai', 'visionai', 'ai2019', 'dsho', 'vthirteenai',
            'V13Ai', 'Thirteenvision', 'Alphavision13', '2019ai', '5ai', 'AIVision13', 'xavai', 'ngthon13', 'songAI', 'prust13ai', 'kyle13', 'wolfAI',
            'CottAI13', 'MinAI', 'Sami13', 'PradVision13']
        free_codes = ['NCT4FREE']
        code = self.ids['discountCode'].text
        if code in discount_codes:
            self.ids['discountCodeValidity'].text = 'Valid'
            self.apply_code = True
            variables.code_applied = True
        elif code in free_codes:
            self.ids['discountCodeValidity'].text = '100% applied'
            self.free = True
            variables.free = True
            self.pay()
            # content = Button(text='OK', size_hint=(1, .1))
            # popup = Popup(title='Please make sure to check terms of service', content=content, size_hint=(.5, .5))
            # content.bind(on_press=popup.dismiss)
            # popup.open()
        else:
            self.ids['discountCodeValidity'].text = 'Invalid'

    def pay(self):
        # Square payment
        if os.path.exists('terms.txt'):
            f = open('terms.txt', 'r')
            try:
                agree = json.load(f)['agreed']
            except:
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
            return
        content = Button(text='OK', size_hint=(1, .1))
        popup = Popup(title='100% discount applied. Feel free to use our service!', content=content, size_hint=(.5, .5))
        content.bind(on_press=popup.dismiss)
        popup.open()
        self.parent.current = 'main'

    def change_paid_status(self):
        to_write = {"paid": True}
        with open('payment.txt', 'w') as f:
            f.seek(0)
            json.dump(to_write, f)


    def change_to_crypto(self):
        if self.free:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='100% discount applied. Feel free to use our service!', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()
            self.parent.current = 'main'
            return
        if os.path.exists('terms.txt'):
            f = open('terms.txt', 'r')
            try:
                agree = json.load(f)['agreed']
            except Exception:
                agree = False
            f.close()
        else:
            return
        print(agree)
        if not agree:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='You must agree to terms of service', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()
            return
        self.parent.current = 'payCrypto'

    def save_order_id(self, order_id):
        data = {}
        data['sq_id'] = order_id
        data['txn_id'] = ''
        data['paid'] = False
        if os.path.exists('payment.txt'):
            with open('payment.txt', 'r+') as f:
                data = json.load(f)
                if 'paid' in data.keys() and data['paid']:
                    return
        else:
            with open('payment.txt', 'w') as f:
                json.dump(data, f)

    def check_order_id(self):
        order_id = ''
        txn_id = ''
        paid_status = ''
        with open('payment.txt', 'r') as json_file:
            data = json.load(json_file)
            try:
                order_id = data['order_id']
                txn_id = data['txn_id']
            except:
                paid_status = data['paid']

        if order_id == '' and txn_id == '':
            if paid_status:
                return 'Paid'
            else:
                return "Not Paid"
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
                f.seek(0)
                json.dump(data, f)
            return 'Payment Accepted'

        #Check crypto_payment id
        try:
            order_status = crypto_payment.get_tx_info(txn_id)
        except Exception as e:
            print(str(e))
        if order_status['result']['status'] == 1:
            data['paid'] = True
            with open('payment.txt', 'w') as f:
                json.dump(data, f)
            return 'Payment Accepted'
        else:
            return 'Not Paid'

class PayByCrypto(Screen):
    def __init__(self, **kwargs):
        super(PayByCrypto, self).__init__(**kwargs)
        self.curr2 = ''

    def currency_select(self, currency):
        self.curr2 = currency

    def update_information(self, address, amount):
        self.ids['address'].text = address
        self.ids['amount'].text = amount

    def create_transaction(self):
        email = self.ids['coinEmail'].text
        # Hacky way to share apply_code variable across screens
        apply_code = variables.code_applied
        if apply_code:
            quantity = 20
        else:
            quantity = 25

        if len(email) > 0:
            res = crypto_payment.create_transaction('USD', self.curr2, quantity, email)
            address = res['result']['address']
            amount = res['result']['amount']
            self.update_information(address, amount)
            id = res['result']['txn_id']
            data = {
                'sq_id': '',
                'txn_id': id,
                'paid': False
            }
            f = open('payment.txt', 'w')
            json.dump(data, f)
            f.close()
        else:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check email and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

class CredentialCheckCrypto(Screen):
    def __init__(self, **kwargs):
        super(CredentialCheckCrypto, self).__init__(**kwargs)
        self.exchange = ''
        self.exchanges = []
        self.currently_available = ['binance', 'binance.us']
        self.part = Particle_publish()

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
        if os.path.exists('crypto.txt'):
            with open('crypto.txt', 'r+') as json_file:
                try:
                    f_data = json.load(json_file)
                except:
                    f_data = []
        else:
            f_data = []
        for i in self.exchanges:
            if i not in f_data:
                f_data.append(i)
        with open('crypto.txt', 'w') as json_file:
            json.dump(f_data, json_file)
        self.manager.current = "main"

    def add(self):
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
        self.exchanges.append(data)

    def handle_crypto_choice(self):
        self.write_file()

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

class CredentialCheckStock(Screen):
    def __init__(self, **kwargs):
        super(CredentialCheckStock, self).__init__(**kwargs)
        self.exchange = ''
        self.exchanges = []
        self.currently_available = ['Robinhood']
        self.part = Particle_publish()

    def on_spinner_select_exchange(self, stock_exchange):
        self.exchange = stock_exchange

    def add(self):
        data = {}
        data['username'] = self.ids['CrChStockName'].text
        data['password'] = self.ids['CrChStockPassw'].text
        data['exchange'] = self.exchange
        if data['exchange'] not in self.currently_available:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Exchange currently unavailable. Support will arrive in 24-48 hours', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            self.part.auth()
            print(self.part.publish_event(event_name='Exchange_request', data=self.exchange))
            popup.open()
        self.exchanges.append(data)

    def write_file(self):
        data = {}
        data['username'] = self.ids['CrChStockName'].text
        data['password'] = self.ids['CrChStockPassw'].text
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
        with open('stock.txt', 'r+') as json_file:
            try:
                f_data = json.load(json_file)
            except:
                f_data = []
        for i in self.exchanges:
            if i not in f_data:
                f_data.append(i)
        with open('stock.txt', 'w') as json_file:
            json.dump(f_data, json_file)
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
        Clock.schedule_once(self.prefill_forex, 1)

    ## Forex API

    def prefill_forex(self, dt):
        if os.path.exists('forex.txt'):
            with open('forex.txt', 'r') as f:
                try:
                    data = json.load(f)
                except:
                    data = None
            if data is not None:
                self.ids['funame'].text = data['Funame']
                self.ids['fpassw'].text = data['Fpassw']
                self.ids['appkey'].text = data['Fappkey']

    def loginForex(self):
        funame = self.ids['funame'].text
        fpassw = self.ids['fpassw'].text
        appkey = self.ids['appkey'].text
        if self.fx.login(funame, fpassw, appkey):
            if not os.path.exists('forex.txt'):
                with open('forex.txt', 'w') as f:
                    json.dump({'Funame': funame, 'Fpassw': fpassw, 'Fappkey': appkey}, f)
            self.manager.current = 'forexPort'
        else:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please check credentials and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

    ## Crypto API

    def on_spinner_select(self, exch):
        self.exchange = exch
        self.prefill_crypto(exch)

    def prefill_crypto(self, exch):
        if os.path.exists('crypto.txt'):
            with open('crypto.txt', 'r') as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(e)
                    data = None
            if data is not None:
                for i in data:
                    print(i)
                    try:
                        saved_exch = i['exchange']
                    except:
                        saved_exch = None
                    print(saved_exch)
                    if saved_exch == exch:
                        self.ids['CAPIKey'].text = i['apiKey']
                        self.ids['CSecretKey'].text = i['secretKey']
                        return
                return


    def loginCrypto(self):
        apikey = self.ids["CAPIKey"].text
        skey = self.ids["CSecretKey"].text
        if self.exchange == '' or apikey =='' or skey == '':
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Please enter all details and try again', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()
        else:
            if self.exchange == 'binance' or self.exchange == 'binance.us':
                bin = Binance(apikey, skey)
                result = bin.account_information()[0]
            else:
                result = 'error'
            if result == 'error':
                content = Button(text='OK', size_hint=(1, .1))
                popup = Popup(title='Please check details and try again', content=content, size_hint=(.5, .5))
                content.bind(on_press=popup.dismiss)
                popup.open()
            else:
                variables.current_crypto = {'api': apikey, 'secret': skey, 'exchange': self.exchange}
                if os.path.exists('crypto.txt'):
                    with open('crypto.txt', 'r+') as f:
                        data = json.load(f)
                        data.append(variables.current_crypto)
                        f.seek(0)
                        json.dump(data, f)
                else:
                    with open('crypto.txt', 'w') as f:
                        json.dump([variables.current_crypto], f)
                cryptoPort = self.manager.get_screen('cryptoPort')
                cryptoPort.balance_information()
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

    def balance_information(self):
        if variables.current_crypto['exchange'] == 'binance' or variables.current_crypto['exchange'] == 'binance.us':
            bin = Binance(variables.current_crypto['api'], variables.current_crypto['secret'])
            nonzero = bin.get_nonzero_currency()
            i = 0
            total = 0
            while i < 6:
                self.ids['cryptoCurr{}'.format(i)].text = nonzero[i]['asset']
                self.ids['curr{}Amount'.format(i)].text = nonzero[i]['free']
                price = round(float(bin.price_ticker(nonzero[i]['asset'] + 'USD')['price']) * float(nonzero[i]['free']), 2)
                total += price
                self.ids['curr{}usd'.format(i)].text = str(price)
                if i == len(nonzero) - 1:
                    break
                i += 1
            self.ids['TotalUSD'].text = str(total) + ' USD'
        else:
            self.ids['cryptoCurr2'].text = 'Information unavailable'

    def buy(self):
        curr_pair = self.ids['currencyPair'].text.upper()
        amount = int(self.ids['Amount'].text)
        current_info = variables.current_crypto
        if current_info['exchange'] == 'binance':
            bin = Binance(current_info['api'], current_info['secret'])
            bin.place_order('BUY', curr_pair, amount)
        else:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Currently unsupported', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

    def sell(self):
        curr_pair = self.ids['currencyPair'].text.upper()
        amount = int(self.ids['Amount'].text)
        current_info = variables.current_crypto
        if current_info['exchange'] == 'binance':
            bin = Binance(current_info['api'], current_info['secret'])
            bin.place_order('SELL', curr_pair, amount)
        else:
            content = Button(text='OK', size_hint=(1, .1))
            popup = Popup(title='Currently unsupported', content=content, size_hint=(.5, .5))
            content.bind(on_press=popup.dismiss)
            popup.open()

    def change_crypto(self):
        new_exchange = self.ids['ExName'].text.lower()
        new_api = self.ids['newApiKey'].text
        new_secret = self.ids['newSecretKey'].text
        current_info = {'api': new_api, 'secret': new_secret, 'exchange': new_exchange}
        variables.current_crypto = current_info
        if new_exchange in variables.accepted_crypto:
            try:
                dispatch = {
                    'binance': Binance(new_api, new_secret),
                    'binance.us': Binance(new_api, new_secret)
                }
                ex = dispatch[new_exchange]
                if ex.account_information()[0] != 'OK':
                    content = Button(text='OK', size_hint=(1, .1))
                    popup = Popup(title='Please check information and try again', content=content, size_hint=(.5, .5))
                    content.bind(on_press=popup.dismiss)
                    popup.open()
            except:
                content = Button(text='OK', size_hint=(1, .1))
                popup = Popup(title='Please check information and try again', content=content, size_hint=(.5, .5))
                content.bind(on_press=popup.dismiss)
                popup.open()

    # def change_crypto(self):
    #     exhange = self.ids['ExName'].text.lower()
    #     if exchange == '' or exchange not in ccxt.exchanges:
    #         content = Button(text='OK', size_hint=(1, .1))
    #         popup = Popup(title='Please check exchange name and try again', content=content, size_hint=(.5, .5))
    #         content.bind(on_press=popup.dismiss)
    #         popup.open()
    #     self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, exchange)
    #     self.ex = exchange

    # def account_balance(self):
    #     self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
    #     bal = crypto.get_account_balance(self.acc)
    #     if bal == 'error':
    #         content = Button(text='OK', size_hint=(1, .1))
    #         popup = Popup(title='Please check exchange and try again', content=content, size_hint=(.5, .5))
    #         content.bind(on_press=popup.dismiss)
    #         popup.open()
    #     grid = GridLayout(cols=1, rows=len(bal.keys()))
    #     dismiss_but = Button(text="Dismiss", size_hint=(1, .1))
    #     for item in bal:
    #         grid.add_widget(Label(text=item + ': ' + str(bal[item])))
    #     grid.add_widget(dismiss_but)
    #     popup = Popup(title='Account Information', content=grid, size_hint=(.5, .5))
    #     dismiss_but.bind(on_press=popup.dismiss)
    #     popup.open()

    # def sell_market(self):
    #     if self.acc is None:
    #         self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
    #     amount = int(self.ids['AmountMarket'].text)
    #     market_id = self.ids['MarketIdMarket'].text
    #     x = crypto.sell_order_market(self.acc, market_id, amount)
    #     if x == 'error':
    #         content = Button(text='OK', size_hint=(1, .1))
    #         popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
    #         content.bind(on_press=popup.dismiss)
    #         popup.open()
    #
    # def buy_market(self):
    #     if self.acc is None:
    #         self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
    #     amount = int(self.ids['AmountMarket'].text)
    #     market_id = self.ids['MarketIdMarket'].text
    #     x = crypto.buy_order_market(self.acc, market_id, amount)
    #     if x == 'error':
    #         content = Button(text='OK', size_hint=(1, .1))
    #         popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
    #         content.bind(on_press=popup.dismiss)
    #         popup.open()
    #
    # def sell_limit(self):
    #     if self.acc is None:
    #         self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
    #     amount = int(self.ids['AmountLimit'].text)
    #     price = int(self.ids['PriceLimit'].text)
    #     market_id = self.ids['MarketIdLimit'].text
    #     x = crypto.sell_order_limit(self.acc, market_id, amount, price)
    #     if x == 'error':
    #         content = Button(text='OK', size_hint=(1, .1))
    #         popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
    #         content.bind(on_press=popup.dismiss)
    #         popup.open()
    #
    # def buy_limit(self):
    #     if self.acc is None:
    #         self.acc = crypto.gen_account_id(crypto.apikey, crypto.skey, crypto.exchange_name)
    #     amount = int(self.ids['AmountLimit'].text)
    #     price = int(self.ids['PriceLimit'].text)
    #     market_id = self.ids['MarketIdLimit'].text
    #     x = crypto.buy_order_limit(self.acc, market_id, amount, price)
    #     if x == 'error':
    #         content = Button(text='OK', size_hint=(1, .1))
    #         popup = Popup(title='Please check market and try again', content=content, size_hint=(.5, .5))
    #         content.bind(on_press=popup.dismiss)
    #         popup.open()



class ForexPortfolio(Screen):
    #DA545354
    def __init__(self, **kwargs):
        super(ForexPortfolio, self).__init__(**kwargs)
        self.quantity = 0
        self.exchange = ''
        self.order_id = ''
        self.credentials = self.check_creds()
        if self.credentials is not None:
            Clock.schedule_once(self.display, 1)

    def check_creds(self):
        if os.path.exists('forex.txt'):
            with open('forex.txt', 'r') as f:
                try:
                    data = json.load(f)
                except:
                    data = None
            return data
        else:
            return

    def display(self, dt):
        fx.login(self.credentials['Funame'], self.credentials['Fpassw'], self.credentials['Fappkey'])
        data = fx.get_margins()
        self.ids['ForexCash'].text = str(data['Cash'])
        self.ids['ForexEquity'].text = str(data['NetEquity'])
        self.ids['ForexTradable'].text = str(data['TradableFunds'])
        self.ids['ForexMargin'].text = str(data['Margin'])
        self.ids['ForexCurrency'].text = str(data['Currency'])

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
        order_id = fx.trade_order(market_id, qnty, 'buy')

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
        # row = self.create_header_current()
        # for item in row:
            # wid = Label(text=item, size_hint=(.167, .2))
            # self.add_widget(wid)
        # for i in range(10):
        #     row = self.create_coin_current(i)
        #     for item in row:
        #         wid = Label(text=item, size_hint=(.167, .2))
        #         self.add_widget(wid)
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


class ForexGrid(GridLayout):
    def __init__(self, **kwargs):
        super(MyGrid, self).__init__(**kwargs)
        self.fetch_data_from_database()
        self.display_data()

    def fetch_data_from_database(self):
        pass



kv = Builder.load_file("gui.kv")


class MyMainApp(App):
    apply_code = False
    def build(self):
        return kv


if __name__ == "__main__":
    MyMainApp().run()

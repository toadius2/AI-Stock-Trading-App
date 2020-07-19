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
from kivy.properties import ObjectProperty
import requests
import re
import requests
import json
import os



class MainWindow(Screen):
    pass

class CreateAccountWindow(Popup):
    uname = ObjectProperty(None)
    passw = ObjectProperty(None)
    email = ObjectProperty(None)
    cpass = ObjectProperty(None)
    def  submit(self):
        self.validate_signUp()
        pass

    def login(self):
        self.reset()
        sm.current = "main"

    def reset(self):
        self.email.text = ""
        self.passw.text = ""
        self.uname.text = ""

    def validate_signUp(self):
        name_signup = self.uname
        email_signup = self.email
        password_signup = self.passw
        confpass_signup = self.cpass
        
        error_signup = " "

        if (re.match('^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$', str(email_signup))
            and (re.match("^[a-zA-Z]+(\s[a-zA-Z]+)?$", name_signup)) and password_signup != "" and confpass_signup != " "):
            if (password_signup != confpass_signup):
                
                error_signup = "error"
                Popup(title='Invalid Form',
                  content=Label(text="Confirm password doesn't match with password entered."),
                  size_hint=(None, None), size=(400, 400))
            
            if(error_signup == " "):
                result = NCTdatabase.createDatabase(name_signup, email_signup, password_signup, dob_signup)
                #print(result)
                if(result != "Error"):
                    self.on_close()
                else:
                    
                    Popup(title='Invalid Form',
                        content=Label(text='Oops! Email already taken.'),
                        size_hint=(None, None), size=(400, 400))

        else:
            
            self.invalidForm()
            return False

    def invalidForm(self):
        pop = Popup(title='Invalid Form',
                  content=Label(text='Please fill in all inputs with valid information.'),
                  size_hint=(None, None), size=(400, 400))
        pop.open()
    def on_close(self):
        self.destroy()
        

class SecondWindow(Screen):
    pass

class CryptoPorfolio(Screen):
    pass

class WindowManager(ScreenManager):
    pass

class NCTTabbedPanel2(TabbedPanel):
    pass

class NCTTabbedPanel(TabbedPanel):
    pass

class NCTTabbedPanel1(TabbedPanel):
    pass

class TableHeader(Label):
    pass

class CoinRecord(Label):
    pass

class MyGrid1(GridLayout):
    def __init__(self, **kwargs):
        super(MyGrid1, self).__init__(**kwargs)
        self.fetch_data_from_marketCap()
        self.display_current_data()

    def fetch_data_from_marketCap(self):
        api_request = requests.get("https://api.coinmarketcap.com/v1/ticker/")
        api = json.loads(api_request.content)
        self.data1 = []
        self.data1 = api

    def display_current_data(self):
        for i in range(15):
            if i < 1:
                row = self.create_header_current(i)
            else:
                row = self.create_coin_current(i)
            for item in row:
                self.add_widget(item)

    def create_header_current(self, i):
        first_column = TableHeader(text='Name')
        second_column = TableHeader(text='Rank')
        third_column = TableHeader(text='Price')
        fourth_column = TableHeader(text='Change1hr')
        fifth_column = TableHeader(text='Change24hr')
        sixth_column = TableHeader(text='Change7d')
        return [first_column, second_column, third_column, fourth_column, fifth_column, sixth_column]

    def create_coin_current(self, i):
        first_column = CoinRecord(text=self.data1[i]['symbol'])
        second_column = CoinRecord(text=self.data1[i]['rank'])
        third_column = CoinRecord(text=self.data1[i]['price_usd'])
        fourth_column = CoinRecord(text=self.data1[i]['percent_change_1h'])
        fifth_column = CoinRecord(text=self.data1[i]['percent_change_24h'])
        sixth_column = CoinRecord(text=self.data1[i]['percent_change_7d'])
        return [first_column, second_column, third_column, fourth_column, fifth_column, sixth_column]


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
                self.add_widget(item)

    def create_header(self, i):
        first_column = TableHeader(text='Name')
        second_column = TableHeader(text='Rank')
        third_column = TableHeader(text='Current Price')
        fourth_column = TableHeader(text='Price Paid')
        fifth_column = TableHeader(text='Profit/Loss')
        sixth_column = TableHeader(text='1 HR change')
        seventh_column = TableHeader(text='24 HR change')
        eighth_column = TableHeader(text='7day change')
        nineth_column = TableHeader(text='Current value')
        return [first_column, second_column, third_column, fourth_column, fifth_column, sixth_column, seventh_column,
                eighth_column, nineth_column]

    def create_coin_info(self, i):
        first_column = CoinRecord(text=self.data[i]['Name'])
        second_column = CoinRecord(text=self.data[i]['Rank'])
        third_column = CoinRecord(text=self.data[i]['Current Price'])
        fourth_column = CoinRecord(text=self.data[i]['Price Paid'])
        fifth_column = CoinRecord(text=self.data[i]['Profit/Loss'])
        sixth_column = CoinRecord(text=self.data[i]['1 HR change'])
        seventh_column = CoinRecord(text=self.data[i]['24 HR change'])
        eighth_column = CoinRecord(text=self.data[i]['7day change'])
        nineth_column = CoinRecord(text=self.data[i]['Current value'])
        return [first_column, second_column, third_column, fourth_column, fifth_column, sixth_column, seventh_column,
                eighth_column, nineth_column]


    

kv = Builder.load_file("mymain_GUI1.kv")


class MyMainApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    MyMainApp().run()

from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


Builder.load_file("TabbedPortfolio.kv")


class NCTTabbedPannel(TabbedPanel): # Root widget
   pass


class TabbedPanelApp(App):
   def build(self):
       return NCTTabbedPannel()

class TableHeader(Label):
    pass

class CoinRecord(Label):
    pass

class MyGrid(GridLayout):

    def __init__(self, **kwargs):
        super(MyGrid, self).__init__(**kwargs)
        self.fetch_data_from_database()
        self.display_data()

    def fetch_data_from_database(self):
        # using dummy data
        self.data = [
            {'Name': 'Bitcoin', 'Rank': '1',"Current Price": '111', "Price Paid":'99', 'Profit/Loss':'23', '1 HR change':'43', '24 HR change':'100', '7day change':'1000', 'Current value':'1567'},
            {'Name': 'Ripple', 'Rank':'2', "Current Price":'111', "Price Paid":'99', 'Profit/Loss':'23', '1 HR change':'43',
             '24 HR change':'100', '7day change':'1000', 'Current value':'1567'},
            {'Name': 'Litcoin', 'Rank':'3', "Current Price":'111', "Price Paid":'99', 'Profit/Loss':'23', '1 HR change':'43',
             '24 HR change':'100', '7day change':'1000', 'Current value':'1567'},
            {'Name': 'EOS', 'Rank':'4', "Current Price":'111', "Price Paid":'99', 'Profit/Loss':'23', '1 HR change':'43',
             '24 HR change':'100', '7day change': '1000', 'Current value': '1567'},
        ]

    def display_data(self):
        self.create_header(0) # create 0th row as the header row

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
        return [first_column, second_column, third_column, fourth_column, fifth_column, sixth_column, seventh_column, eighth_column,nineth_column]

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


if __name__ == '__main__':
   TabbedPanelApp().run()

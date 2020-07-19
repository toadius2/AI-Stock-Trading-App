# import the GUI package Kivy
import kivy
from kivy.properties import NumericProperty
from kivy.properties import StringProperty
from kivy.uix.scatter import Scatter
from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.clock import Clock, mainthread
from kivy.uix.gridlayout import GridLayout
from collections import defaultdict
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.widget import Widget


# class MyApp(App):
#     def build(self):
#         return Label(text="example")

class ScreenManagement(ScreenManager):
    pass

class MainScreen(Screen):
    pass

class PerfectTrade(Screen):
    pass

class Arbitrage(Screen):
    pass

class Robinhood(Screen):
    pass

class PullPortfolio(Screen):
    pass

class Sentiment(Screen):
    pass

class ScreenTwo(Screen):
    pass

# class MyGrid(Widget):
#     pass
#
# #execute run method from App class
# NCTCryptoApp().run()
#
#
# print("Hello World")

presentation = Builder.load_file("portfolio.kv")
class NCTCryptoV2(App):

    def build(self):
        return presentation


NCTCryptoV2().run()

if __name__ == "__main__":
    NCTCryptoV2().run()

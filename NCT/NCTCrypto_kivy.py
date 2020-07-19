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

# The login or start page
class MainWindow(Screen):
    pass

# Page after successful login
class SecondWindow(Screen):
    pass

# Transition between the windows
class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("guiscreen.kv")

class CryptoPortfolioApp(App):
    def build(self):
        return kv

if __name__ == "__main__":
    CryptoPortfolioApp().run()

import json
from tkinter import *
import tkinter as tk
from tkinter import ttk
import ccxt as ccxt
import requests
from tkinter import messagebox
# import 3 api classes for the 3 tabs
import Forex_older as Fx
import data_mongo as dm
#import Robinhood_new as Robinhood

def raise_frame(frame):
    frame.tkraise()

def clean_up(frame):
    forex_portfolio_login_screen.my_trader = None
    frame.tkraise()

def on_closing():
    tk.messagebox.showinfo("Info", "Closing the window")
    # TODO: Log-off from Robinhood and CCXT
    if forex_portfolio_login_screen.my_trader is not None:
        # delete extra rows if more than 5
        dm.update_collection(forex_portfolio_login_screen.my_trader.username)  # added
        print("Forex database updated.")
        forex_portfolio_login_screen.my_trader.log_off()
        # Print data from database
        #result = dm.fetch_forex_data(forex_portfolio_login_screen.my_trader.username)
    root.destroy()


root = tk.Tk()
root.geometry("2060x720")
root.configure(background='black')
root.title("NCT Software")
frame1 = Frame(root)
frame1.configure(background='black')
Screen2 = Frame(root)
Screen3 = Frame(root)
Screen4 = Frame(root)
Screen5 = Frame(root)
for frame in (frame1, Screen2, Screen3, Screen4, Screen5):
    frame.grid(row=0, column=0, sticky="news")
class MyPortfolio():
    def __init__(self,myportfolio):
        label = Label(myportfolio, text="                      LOGIN ", font="Arial 10 bold", width=15, height=3) \
            .grid(row=1, column=3, sticky="sw")
        blank = Label(myportfolio, text=" ", width=15).grid(row=2, column=1)
        label = Label(myportfolio, text="       Enter Username : ", width=20, height = 2, pady=5, padx=5).grid(row=3, column=2, sticky = "w")
        user_box = Entry(myportfolio, width = 30, borderwidth=3).grid(row=3, column=3, columnspan = 2)
        label = Label(myportfolio, text="       Enter Password : ", width=20, height = 3)
        label.grid(row=5, column=2, sticky = "n")
        password_box = Entry(myportfolio,show='*', width = 30, borderwidth=3).grid(row=5, column=3, columnspan = 2)
        btn = Button(myportfolio, text="Login", command=lambda: raise_frame(Screen2), width=15).grid(row=6, column=3, sticky= "w")
        blank_button = Label(myportfolio, text= " ").grid(row=7, column=3, sticky="w")
        btn_signup = Button(myportfolio, text="SignUp", command=lambda:self.signUp() , width=15)
        btn_signup.grid(row=8, column=3, sticky="w")
    def signUp(self):
        popUp(self)
class popUp(Toplevel):
    def __init__(self, original):
        Toplevel.__init__(self)
        self.geometry("550x350+350+230")
        self.title("SignUp")
        label = Label(self, text = "   SignUp ",height=2,font="Arial 15 bold")
        label.grid(row=0,column=3,sticky='w')
        label = Label(self, text=" Name : ",height=2)
        label.grid(row=1, column=2, sticky='w')
        namebox = Entry(self,width=30, borderwidth=3).grid(row=1,column=3)
        label = Label(self, text=" UserName : ",height=2)
        label.grid(row=2, column=2, sticky='w')
        ubox = Entry(self,width=30,borderwidth=3).grid(row=2,column=3)
        label = Label(self, text=" Password : ",height=2)
        label.grid(row=3, column=2, sticky='w')
        passbox = Entry(self,show="*",width=30, borderwidth=3).grid(row=3,column=3)
        label = Label(self, text=" Confirm Password : ",height=2)
        label.grid(row=4, column=2, sticky='w')
        cpassbox = Entry(self,show="*",width=30, borderwidth=3).grid(row=4,column=3)
        label = Label(self, text=" Date Of Birth :(MM/DD/YYYY) ",height=2)
        label.grid(row=5, column=2, sticky='w')
        dobbox = Entry(self,width=30, borderwidth=3).grid(row=5,column=3)
        btn = Button(self, text ="SignUp", command= lambda : self.on_close(),height=1,width=10)
        btn.grid(row =7,column=3, sticky='w')
    def on_close(self):
        self.destroy()
        root.update()
        root.deiconify()
        raise_frame(Screen2)


class crypto_protfolio_page():
    def __init__(self, Screen3):
        current_balance = Label(Screen3, text="Current Balance :", font="Times 9 bold", height=2)
        current_balance.grid(row=1, column=0, sticky="news")
        header_name = Label(Screen3, text="Name", bg="white", font="Times 9 bold", width=15)
        header_name.grid(row=2, column=0, sticky="news")
        header_Rank = Label(Screen3, text="Rank   ", bg="light grey", font="Times 9 bold", width=15)
        header_Rank.grid(row=2, column=1, sticky="news")
        header_Current_Price = Label(Screen3, text="Current Price", bg="white", font="Times 9 bold", width=15)
        header_Current_Price.grid(row=2, column=2, sticky="news")
        header_price_paid = Label(Screen3, text="Price Paid ", bg="light grey", font="Times 9 bold", width=15)
        header_price_paid.grid(row=2, column=3, sticky="news")
        header_current_value = Label(Screen3, text="Current Market Price", bg="white", font="Times 9 bold", width=15)
        header_current_value.grid(row=2, column=4, sticky="news")
        header_profit_loss = Label(Screen3, text="Profit/Loss  ", bg="light grey", font="Times 9 bold", width=15)
        header_profit_loss.grid(row=2, column=5, sticky="news")
        btn_Crypto_portfolio = Button(Screen3, text="Go back", command=lambda: raise_frame(Screen2), width = 20)
        btn_Crypto_portfolio.grid(row=15, column=1, sticky = "s")
        btn_Crypto_portfolio = Button(Screen3, text="Subbscribe/Unsubscribe to particle Board", command=lambda: raise_frame(Screen2), width=40)
        btn_Crypto_portfolio.grid(row=15, column=2, columnspan= 3,  sticky="s")
class popUp_CryptoPortfolio(Toplevel):
    def __init__(self, original):
        Toplevel.__init__(self)
        self.geometry("450x300+350+230")
        self.title("SignIn Crypto Portfolio")
        label = Label(self, text=" SignIn ", font = "Arial 10 bold", height = 5)
        label.grid(row=0, column=3)
        label = Label(self, text=" Enter API Key : ", height = 2)
        label.grid(row=1, column=2)
        apibox = Entry(self, width = 30, borderwidth = 3).grid(row=1, column=3)
        label = Label(self, text=" Enter Secret key : ", height = 2)
        label.grid(row=2, column=2)
        secretbox = Entry(self,show='*', width = 30, borderwidth = 3).grid(row=2, column=3)
        btn = Button(self, text="Submit", command=lambda: self.on_close())
        btn.grid(row=4, column=3)

    def on_close(self):
        self.destroy()
        root.update()
        root.deiconify()
        raise_frame(Screen3)


class crypto_login_screen():
    def __init__(self, crptoportfolio):
        tkvar = StringVar(crptoportfolio)
        crypto_exchanges = ccxt.exchanges
        #print(crypto_exchanges)
        choices = crypto_exchanges
        tkvar.set('binance')
        popupMenu = OptionMenu(crptoportfolio, tkvar, *choices)
        Label(crptoportfolio, text=" ", width=40, height = 5).grid(row=4, column=1)
        Label(crptoportfolio, text="Choose Exchange", width=20, height=5).grid(row=5, column=2, rowspan=2, sticky="e")
        popupMenu.grid(row=5, column=3, rowspan=2, columnspan=2)
        signup = Button(crptoportfolio, text="  Submit   ", font="none, 10", width = 12, command=lambda: self.signUp()) \
            .grid(row=7, column=4)
        Label(crptoportfolio, text=" ", width=38, height = 14).grid(row=8, column=5)
        btn_crypto_login = Button(crptoportfolio, text="Go back", command=lambda: raise_frame(frame1),
                                  width=15)
        btn_crypto_login.grid(row=10, column=6, sticky="se")
        # on change dropdown value
        def change_dropdown(*args):
            print(tkvar.get())
        # link function to change dropdown
        tkvar.trace('w', change_dropdown)
    def signUp(self):
        popUp_CryptoPortfolio(self)
class forex_portfolio_login_screen():
    # added
    my_trader: Fx.Forex_trader = None
    username = ""
    password = ""
    auth_token = ""
    def __init__(self,forex_portfolio):

        label = Label(forex_portfolio, text=" LOGIN ").grid(row=1,column=3)
        label = Label(forex_portfolio, text=" Enter Username : ")
        label.grid(row=2, column=2)
        self.uname = tk.StringVar()
        self.forex_username = Entry(forex_portfolio, textvariable=self.uname).grid(row=2, column=3)

        label = Label(forex_portfolio, text=" Enter Password : ")
        label.grid(row=3, column=2)

        self.pwd = tk.StringVar()
        forex_password = Entry(forex_portfolio, textvariable=self.pwd, show='*').grid(row=3, column=3)

        label = Label(forex_portfolio, text=" Enter App-Key : ")  # added
        label.grid(row=4, column=2)
        self.appKey = tk.StringVar()
        forex_appkey = Entry(forex_portfolio, textvariable=self.appKey).grid(row=4, column=3)

        # added
        btn_submit = Button(forex_portfolio, text="Submit",command=lambda: self.validate_input(self.uname.get(),self.pwd.get(), self.appKey.get())).grid(row=5, column=3)
        #self.btn = Button(forex_portfolio, text="Submit",command=self.validate_input).grid(row=5, column=3)
        btn_forex_login = Button(forex_portfolio, text="Go back", command=lambda: raise_frame(frame1), width=20)
        btn_forex_login.grid(row=6, column=2, sticky="s")

    def validate_input(self, uname, pwd, appkey):  # added appkey
        print("SELF Validate Input")
        # validate for blank strings in any of the fields before API call
        if uname == "" or pwd == "" or appkey == "":  # added
            tk.messagebox.showwarning("Error", "All fields are required!")  # added

        elif forex_portfolio_login_screen.my_trader is None:  # added elif to if
            # Login to Forex account with given uname, pwd
            forex_portfolio_login_screen.my_trader = Fx.Forex_trader(uname, pwd, appkey)  # added appkey

            if forex_portfolio_login_screen.my_trader.login():
                print("Login successful!/n")
                forex_portfolio_login_screen.username = uname
                print("Session id = ", forex_portfolio_login_screen.my_trader.auth_token)
                # Goto portfolio page
                forex_portfolio_page(Screen4, uname, forex_portfolio_login_screen.my_trader.auth_token)

                #raise_frame(Screen4)
            else:
                forex_portfolio_login_screen.my_trader = None
                print("Unable to login!")
                tk.messagebox.showwarning("Error","Username/Password do not match!")
                # Set the Password input box to blank
                # self.uname.set("")
                self.pwd.set("")
                #raise_frame(frame1)
        else:
            # Goto portfolio page
            forex_portfolio_page(Screen4, uname, forex_portfolio_login_screen.my_trader.auth_token)
            #raise_frame(Screen4)


def validate_input(uname, pwd):
    print("********************  Username is {} and Password if {} ".format(uname, pwd))
    #raise_frame(Screen4, uname, pwd)
    forex_portfolio_page(Screen4, uname, pwd)

def qf(param):
    print(param)

class robinhood_login_screen():
    def __init__(self,robinhood_portfolio):
        label = Label(robinhood_portfolio, text=" LOGIN ", font = "Arial 10 bold", width = 10, height = 7)\
            .grid(row=1, column=2, sticky = "e")
        blank = Label(robinhood_portfolio, text=" ", width=30).grid(row = 2, column = 1)
        label = Label(robinhood_portfolio, text=" Enter Username : ", width = 30)
        label.grid(row=3, column=2)
        robin_name_box = Entry(robinhood_portfolio).grid(row=3, column=3)
        label = Label(robinhood_portfolio, text=" Enter Password : ", width = 30)
        label.grid(row=4, column=2)
        robin_password_box = Entry(robinhood_portfolio,show='*').grid(row=4, column=3)
        btn = Button(robinhood_portfolio, text="Submit",command=lambda :self.signUp_token_robinhood())\
            .grid(row=5, column=3)
        btn_robinhood_login = Button(robinhood_portfolio, text="Go back", command=lambda: raise_frame(frame1), width=20)
        btn_robinhood_login.grid(row=5, column=2, sticky="s")
    def signUp_token_robinhood(self):
        popUp_token_robinhoodPortfolio(self)
class popUp_token_robinhoodPortfolio(Toplevel):
    def __init__(self, original):
        self.original_frame = original
        Toplevel.__init__(self)
        self.transient(Screen2)
        self.geometry("350x150+350+230")
        self.title("Enter Token for Authentication")
        self.lift()
        label = Label(self, text=" Enter Token : ")
        label.grid(row=1, column=2)
        token_robinhood_box = Entry(self).grid(row=1, column=3)
        btn_robinhood = Button(self, text="Submit", command=lambda: self.on_close())
        btn_robinhood.grid(row=4, column=3)
    def on_close(self):
        self.destroy()
        root.update()
        root.deiconify()
        raise_frame(Screen5)
class robinhood_portfolio_page():
    def __init__(self,Screen5):
        current_balance = Label(Screen5, text="Current Balance :", font="Times 9 bold", height=2)
        current_balance.grid(row=1, column=0, sticky="news")
        header_name = Label(Screen5, text="Name", bg="white", font="Times 9 bold", width=25)
        header_name.grid(row=2, column=0, sticky="news")
        header_Rank = Label(Screen5, text="Rank   ", bg="light grey", font="Times 9 bold", width=25)
        header_Rank.grid(row=2, column=1, sticky="news")
        header_Current_Price = Label(Screen5, text="Current Price", bg="white", font="Times 9 bold", width=25)
        header_Current_Price.grid(row=2, column=2, sticky="news")
        header_price_paid = Label(Screen5, text="Price Paid ", bg="light grey", font="Times 9 bold", width=25)
        header_price_paid.grid(row=2, column=3, sticky="news")
        header_current_value = Label(Screen5, text="Current Market Price", bg="white", font="Times 9 bold", width=25)
        header_current_value.grid(row=2, column=4, sticky="news")
        header_profit_loss = Label(Screen5, text="Profit/Loss  ", bg="light grey", font="Times 9 bold", width=25)
        header_profit_loss.grid(row=2, column=5, sticky="news")
        btn_Robinhood_portfolio = Button(Screen5, text="Go back", command=lambda: raise_frame(Screen2), width=20)
        btn_Robinhood_portfolio.grid(row=15, column=1, sticky="s")
        btn_Robinhood_portfolio = Button(Screen5, text="Subbscribe/Unsubscribe to particle Board",
                                      command=lambda: raise_frame(Screen2), width=40)
        btn_Robinhood_portfolio.grid(row=15, column=2, columnspan=3, sticky="s")


class forex_portfolio_page():
    # added
    my_trader: Fx.Forex_trader

    def __init__(self, Screen4, username, session):
    #def __init__(self, Screen4):
        print("Inside Init page******")
        self.my_trader = forex_portfolio_login_screen.my_trader
        print("Username = ",forex_portfolio_login_screen.username)
        print("password = ", forex_portfolio_login_screen.password)
        print("Session = ", forex_portfolio_login_screen.auth_token)
        raise_frame(Screen4)
        #print("In Portfolio Page +++++++ {} + {} ".format(username, session))
        current_balance = Label(Screen4, text="Current Balance :", font="Times 9 bold", height=2)
        current_balance.grid(row=1, column=0, sticky="news")
        header_name = Label(Screen4, text="Balance", bg="white", font="Times 9 bold", width=25)
        header_name.grid(row=2, column=0, sticky="news")
        header_Rank = Label(Screen4, text="Currency", bg="light grey", font="Times 9 bold", width=25)
        header_Rank.grid(row=2, column=1, sticky="news")
        header_Current_Price = Label(Screen4, text="Date    ", bg="white", font="Times 9 bold", width=25)
        header_Current_Price.grid(row=2, column=2, sticky="news")
        # header_price_paid = Label(Screen4, text="Price Paid ", bg="light grey", font="Times 9 bold", width=25)
        # header_price_paid.grid(row=2, column=3, sticky="news")
        # header_current_value = Label(Screen4, text="Current Market Price", bg="white", font="Times 9 bold",
        #                                  width=25)
        # header_current_value.grid(row=2, column=4, sticky="news")
        # header_profit_loss = Label(Screen4, text="Profit/Loss  ", bg="light grey", font="Times 9 bold", width=25)
        # header_profit_loss.grid(row=2, column=5, sticky="news")
        #btn_forex_portfolio = Button(Screen4, text="Go back", command=lambda: raise_frame(Screen2), width=20)

        btn_forex_portfolio = Button(Screen4, text="Go back", command=lambda: clean_up(Screen2), width=20)
        btn_forex_portfolio.grid(row=15, column=1, sticky="s")
        btn_forex_portfolio = Button(Screen4, text="Subbscribe/Unsubscribe to particle Board",
                                             command=lambda: raise_frame(Screen2), width=40)
        btn_forex_portfolio.grid(row=15, column=2, columnspan=3, sticky="s")

        # Adding data for DA545354/Forex123 account
        result = self.my_trader.get_client_margin()
        if result['IsError']:
            print("Error occurerd in getClientMargin()!!!!!!!!!")

        # Un-comment this code and comment the following block of code
        #  to display only the current balance info on screen
        # current_balance_value = Label(Screen4, text=result['Cash'])
        # current_balance_value.grid(row=3, column=0)
        # current_margin = Label(Screen4, text=result['Margin'])
        # current_margin.grid(row=3, column=1)
        # current_currency = Label(Screen4, text=result['Currency'])
        # current_currency.grid(row=3, column=2)

        # Adding the above data in the Database
        insertedId = dm.create_forex_db(self.my_trader.username, result['Cash'], result['Currency'])

        # Fetch all previous records from the database and show on the screen
        n, result = dm.fetch_forex_data(self.my_trader.username)
        for rownum in range(n):
            current_balance_value = Label(Screen4, text=result[rownum]["balance"])
            current_balance_value.grid(row=rownum+3, column=0)
            currency_value = Label(Screen4, text=result[rownum]["currency"])
            currency_value.grid(row=rownum+3, column=1)
            date_value = Label(Screen4, text=result[rownum]["date"])
            date_value.grid(row=rownum+3, column=2)



def financeHome(financehome):
    header_id = Label(financehome, text="ID", bg="light grey", font="Times 9 bold", width=24)
    header_id.grid(row=2, column=0, sticky="news")
    header_name = Label(financehome, text="NAME", bg="white", font="Times 9 bold", width=24)
    header_name.grid(row=2, column=1, sticky="news")
    header_symbol = Label(financehome, text="SYMBOL", bg="light grey", font="Times 9 bold", width=24)
    header_symbol.grid(row=2, column=2, sticky="news")
    header_rank = Label(financehome, text="RANK", bg="white", font="Times 9 bold", width=24)
    header_rank.grid(row=2, column=3, sticky="news")
    header_price_usd = Label(financehome, text="PRICE USD", bg="light grey", font="Times 9 bold", width=24)
    header_price_usd.grid(row=2, column=4, sticky="news")
    header_price_btc = Label(financehome, text="PRICE BTC", bg="white", font="Times 9 bold",
                                 width=24)
    header_price_btc.grid(row=2, column=5, sticky="news")
    header_marketcap_usd = Label(financehome, text="MARKET CAP USD", bg="light grey", font="Times 9 bold",
                             width=24)
    header_marketcap_usd.grid(row=2, column=6, sticky="news")
    resp = requests.get('https://api.coinmarketcap.com/v1/ticker/')
    content = json.loads(resp.content)
    data = []
    data = content
    i=3
    for item in data:
        if(i<16):
            header_id = Label(financehome, text=item['id'], bg="light grey", font="Times 9", width=16,height=2)
            header_id.grid(row=i, column=0, sticky="news")
            header_name = Label(financehome, text=item['name'] ,bg="white", font="Times 9", width=16,height=2)
            header_name.grid(row=i, column=1, sticky="news")
            header_symbol = Label(financehome, text=item['symbol'], bg="light grey", font="Times 9", width=16,height=2)
            header_symbol.grid(row=i, column=2, sticky="news")
            header_rank = Label(financehome, text=item['rank'], bg="white", font="Times 9", width=16,height=2)
            header_rank.grid(row=i, column=3, sticky="news")
            header_price_usd = Label(financehome, text=item['price_usd'], bg="light grey", font="Times 9", width=16,height=2)
            header_price_usd.grid(row=i, column=4, sticky="news")
            header_price_btc = Label(financehome, text=item['price_btc'], bg="white", font="Times 9", width=16,height=2)
            header_price_btc.grid(row=i, column=5, sticky="news")
            header_marketcap_usd = Label(financehome, text=item['market_cap_usd'], bg="light grey", font="Times 9",width=16,height=2)
            header_marketcap_usd.grid(row=i, column=6, sticky="news")
            i+=1
class MainApp():
    # added ************************************
    # Define the Handler for Delete Window protocol
    root.protocol("WM_DELETE_WINDOW", on_closing)
    # *******************************************
    Label(frame1, text = " ", bg = "black", width = 25).grid(row = 0, column = 0)
    img = PhotoImage(file="nct_new.png")
    image1 = tk.Label(frame1, image=img, width=250 , height = 250, bg = "black").grid(row=0, column=1)
    label1 = tk.Label(frame1, text="Welcome to NCT Software", font="Arial 15 bold", fg="light yellow",
                      height=7, width=30, bg ="black"). \
        grid(row=0, column=2, sticky = "w")
    style = ttk.Style()
    ttk.Style().configure("TNotebook.Tab", background="black",
                          foreground="blue", bordercolor="black")
    nb = ttk.Notebook(frame1, width=800, height=350)
    nb.grid(row=2, column=1, columnspan=10, sticky = "w")
    myportfolio = ttk.Frame(nb)
    #style.configure(myportfolio, background='black')
    nb.add(myportfolio, text='    My Portfolio                                                       '
                             '                ')
    # LoginFrame(myportfolio)
    MyPortfolio(myportfolio)
    financehome = ttk.Frame(nb)
    nb.add(financehome, text='    Finance Home                                                         '
                            '                ')
    financeHome(financehome)
    style.configure('TNotebook.Tab', font = "Arial 10 bold")
    # Screen 2
    Label(Screen2, text="My Portfolios", font = "Arial 12 bold", width = 115, bg = "rosybrown", height = 1).grid(row=0, column=0, columnspan=8)
    blank_screen2 = Label(Screen2, text=" ", width=5).grid(row=1, column=0)
    nb2 = ttk.Notebook(Screen2, width=1130, height=580)
    nb2.grid(row=5, column=3, columnspan=3)
    # tab1
    cryptoportfolio = ttk.Frame(nb2)
    nb2.add(cryptoportfolio, text='    Crypto Portfolio                                                 ')
    crypto_login_screen(cryptoportfolio)
    # Screen 3
    Label(Screen3, text="Crypto Portfolio", font="Arial 10 bold ", width=140, bg="rosybrown").grid(row=0, column=0,
                                                                                              columnspan=6)
    #Display Crypto Portfolio page
    crypto_protfolio_page(Screen3)
    # tab2
    forex_portfolio = ttk.Frame(nb2)
    nb2.add(forex_portfolio, text='    Forex Portfolio                                                  ')
    forex_portfolio_login_screen(forex_portfolio)
    Label(Screen4, text="Forex Portfolio", font="Arial 10 bold ", width=140, bg="rosybrown").grid(row=0, column=0,
                                                                                                   columnspan=6)
    #forex_portfolio_page(Screen4)
    # tab3
    robinhood_portfolio = ttk.Frame(nb2)
    nb2.add(robinhood_portfolio, text='    Robinhood Portfolio                                          ')
    robinhood_login_screen(robinhood_portfolio)
    Label(Screen5, text="Robinhood Portfolio", font="Arial 10 bold ", width=140, bg="rosybrown").grid(row=0, column=0,
                                                                                                   columnspan=6)
    robinhood_portfolio_page(Screen5)
if __name__ == "__main__":
    raise_frame(frame1)
    MainApp()
    # # Define the Handler for Delete Window protocol # deleted
    # root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
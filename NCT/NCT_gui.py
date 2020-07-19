import json
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import ccxt as ccxt
from NCTDatabase import *

import requests
import re
import datetime
# import 3 api classes for the 3 tabs
import Forex_trader as Fx
import cryptocurrency

crypto_apikey = ""
crypto_secretekey = ""
crypto_exchange = ""
user_email = ""


def raise_frame(frame):
    frame.tkraise()

def clean_up(frame):
    forex_portfolio_login_screen.my_trader = None
    frame.tkraise()


def on_closing():
    # tk.messagebox.showinfo("Info", "Closing the window")
    # TODO: Log-off from Robinhood and CCXT
    if forex_portfolio_login_screen.my_trader is not None:
        # delete extra rows if more than 5
        NCTdatabase.update_collection(forex_portfolio_login_screen.my_trader.username)  # added
        print("Forex database updated.")
        forex_portfolio_login_screen.my_trader.log_off()
        # Print data from database
        #result = dm.fetch_forex_data(forex_portfolio_login_screen.my_trader.username)
    root.destroy()


root = tk.Tk()
root.geometry("2060x720")
root.configure(background='black')
root.title("NCT Software")
#Frame1 - Main login page for NCT
frame1 = Frame(root)
frame1.configure(background='black')
#Login Screen for Portforlios(Crypto, Forex, RobinHood)
Screen2 = Frame(root)
Screen2.configure(background='black')
#Crypto Portfolio Page
Screen3 = Frame(root)
Screen3.configure(background='black')
#Forex Portfolio Page
Screen4 = Frame(root)
Screen4.configure(background='black')
#Robinhood Portfolio Page
Screen5 = Frame(root)
Screen5.configure(background='black')

for frame in (frame1, Screen2, Screen3, Screen4, Screen5):
    frame.grid(row=0, column=0, sticky="news")

#Login for NCT
class MyPortfolio():
    def __init__(self,myportfolio):
        label = Label(myportfolio, text="                      LOGIN ", font="Arial 10 bold", width=15, height=3) \
            .grid(row=1, column=3, sticky="w")
        blank = Label(myportfolio, text=" ", width=50).grid(row=2, column=1)
        label = Label(myportfolio, text="    Enter Email Id : ", width=20, height = 2, pady=5, padx=5).grid(row=3, column=2, sticky = "w")
        self.userboxcon = tk.StringVar()
        self.user_box = Entry(myportfolio, width = 30, borderwidth=3, textvariable=self.userboxcon)\
            .grid(row=3, column=3, columnspan = 2)
        label = Label(myportfolio, text="       Enter Password : ", width=20, height = 3)
        label.grid(row=5, column=2, sticky = "n")
        self.passboxcon = tk.StringVar()
        self.password_box = Entry(myportfolio,show='*', width = 30, borderwidth=3, textvariable=self.passboxcon).grid(row=5, column=3, columnspan = 2)
        btn = Button(myportfolio, text="Login", command=lambda: self.validate(), width=15).grid(row=6, column=3, sticky= "w")
        blank_button = Label(myportfolio, text= " ").grid(row=7, column=3, sticky="w")
        btn_signup = Button(myportfolio, text="SignUp", command=lambda:self.signUp() , width=15)
        btn_signup.grid(row=8, column=3, sticky="w")
    def signUp(self):
        popUp(self)
    def validate(self):
        #print(self.userboxcon)
        email_login = self.userboxcon.get()
        password_login = self.passboxcon.get()
        global user_email

        if re.match('^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$', email_login):
            #print("Email is valid")
            if (password_login != ""):
                result = NCTdatabase.validateUser(email_login, password_login)
                if(result == "Error"):
                    tk.messagebox.showerror("Error", "Invalid Username/Password")
                else:
                    raise_frame(Screen2)
                    user_email = self.userboxcon.get()
                    self.userboxcon.set("")
                    self.passboxcon.set("")

            else:
                tk.messagebox.showwarning("Error", "Password can't be blank")
                return False
        #       print("Email is invalid")
        #if((self.userboxcon.get().isalpha()) and (self.passboxcon.get()!="")):
        #    raise_frame(Screen2)
        #   return True

        else:
            tk.messagebox.showwarning("Error", "Invalid User Name")
            return False

# Create/SignUp Screen for NCT
class popUp(Toplevel):
    def __init__(self, original):
        Toplevel.__init__(self)
        self.geometry("550x350+350+230")
        self.title("SignUp")
        label = Label(self, text = "   SignUp ",height=2,font="Arial 15 bold")
        label.grid(row=0,column=3,sticky='w')
        label = Label(self, text=" Name : ",height=2)
        label.grid(row=1, column=2, sticky='w')
        self.nameboxcon = tk.StringVar()
        namebox = Entry(self,width=30, borderwidth=3, textvariable = self.nameboxcon).grid(row=1,column=3)
        label = Label(self, text=" Email Id : ",height=2)
        label.grid(row=2, column=2, sticky='w')
        self.usernameboxcon = tk.StringVar()
        ubox = Entry(self,width=30,borderwidth=3, textvariable = self.usernameboxcon).grid(row=2,column=3)
        label = Label(self, text=" Password : ",height=2)
        label.grid(row=3, column=2, sticky='w')
        self.passwordboxcon = tk.StringVar()
        passbox = Entry(self,show="*",width=30, borderwidth=3, textvariable = self.passwordboxcon).grid(row=3,column=3)
        label = Label(self, text=" Confirm Password : ",height=2)
        label.grid(row=4, column=2, sticky='w')
        self.confpassboxcon = tk.StringVar()
        cpassbox = Entry(self,show="*",width=30, borderwidth=3, textvariable = self.confpassboxcon).grid(row=4,column=3)
        label = Label(self, text=" Date Of Birth :(MM/DD/YYYY) ",height=2)
        label.grid(row=5, column=2, sticky='w')
        self.dobboxcon = tk.StringVar()
        dobbox = Entry(self,width=30, borderwidth=3, textvariable = self.dobboxcon).grid(row=5,column=3)
        btn = Button(self, text ="SignUp", command= lambda : self.validate_signUp(),height=1,width=10)
        btn.grid(row =7,column=3, sticky='w')
    def on_close(self):
        self.destroy()
        root.update()
        root.deiconify()
        #raise_frame(Screen2)
    def validate_signUp(self):
        name_signup = self.nameboxcon.get()
        email_signup = self.usernameboxcon.get()
        password_signup = self.passwordboxcon.get()
        confpass_signup = self.confpassboxcon.get()
        dob_signup = self.dobboxcon.get()
        error_signup = " "
        if (re.match('^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$', email_signup)
            and (re.match("^[a-zA-Z]+(\s[a-zA-Z]+)?$", name_signup)) and password_signup != "" and confpass_signup != " "
            and dob_signup != " "):
            if (password_signup != confpass_signup):
                self.grab_set()
                error_signup = "error"
                tk.messagebox.showerror("Error", "Confirm password doesn't match with password entered.")
            if(dob_signup != " " and error_signup == " "):
                try:
                    dob_signup = datetime.datetime.strptime(dob_signup, "%m/%d/%Y")
                except:
                    self.grab_set()
                    error_signup = "error"
                    tk.messagebox.showwarning("Error", "Input valid date in mm/dd/yyyy")
            if(error_signup == " "):
                result = NCTdatabase.createDatabase(name_signup, email_signup, password_signup, dob_signup)
                #print(result)
                if(result != "Error"):
                    self.on_close()
                else:
                    self.grab_set()
                    tk.messagebox.showerror("Error", "Oops! Email already taken.")

        else:
            self.grab_set()
            tk.messagebox.showerror("Error", "Input valid data")
            return False

#Crypto Portfolio Page
class crypto_protfolio_page():
    def __init__(self, Screen3):
        global crypto_apikey
        global crypto_secretekey
        global crypto_exchange
        global user_email
        print(crypto_apikey)
        print(crypto_secretekey)
        print(crypto_exchange)
        crypto_balance, crypto_auth = cryptocurrency.get_account_balance(
            cryptocurrency.get_exchange_key_dict(crypto_apikey, crypto_secretekey, crypto_exchange), crypto_exchange)
        # save current balance to data base
        NCTdatabase.saveCurrentBalanceCcxt(user_email, crypto_exchange, crypto_balance)
        header_Current_balance = Label(Screen4, text="Current Balance", bg="black",
                                       fg="white", font="Times 9 bold", width=12)
        header_Current_balance.grid(row=3, column=0)
        header_Current_blank = Label(Screen4, text="    ", bg="black",
                                     fg="white", font="Times 9 bold", width=20)
        header_Current_blank.grid(row=3, column=1)

        header_currency = Label(Screen4, text="Currency   ", bg="black",
                                fg="white", font="Times 9 bold", width=12)
        header_currency.grid(row=3, column=2, sticky="e")
        header_blank = Label(Screen4, text="   ", bg="black",
                             fg="white", font="Times 9 bold", width=20)
        header_blank.grid(row=3, column=3, sticky="news")
        header_date = Label(Screen4, text="Date", bg="black",
                            fg="white", font="Times 9 bold", width=30)
        header_date.grid(row=3, column=4, sticky="news")
        #btn_Crypto_portfolio = Button(Screen3, text="Go back", command=lambda: raise_frame(Screen2), width = 20)
        #btn_Crypto_portfolio.grid(row=15, column=1, sticky = "s")
        btn_Crypto_portfolio = Button(Screen3, text="Activate/Deactivate AI Access Hardware Key", command=lambda: self.crypto_particle_board(), width=45)
        btn_Crypto_portfolio.grid(row=2, column=3, columnspan= 4,  sticky="e")
        crypto_balances_list = NCTdatabase.fetchCcxtCurrentBalnces(user_email,
                                                                   crypto_exchange)


        '''# print(crypto_balances_list)

        for row_num in range(4):
            current_balance_value = Label(Screen3,
                                          text=crypto_balances_list[row_num]["UserExchName"])
            current_balance_value.grid(row=row_num + 3, column=0)
            currency_value = Label(Screen3, text=crypto_balances_list[row_num]["CurrentBalance"])
            currency_value.grid(row=row_num + 3, column=2)
            date_value = Label(Screen3, text=crypto_balances_list[row_num]["date"])
            date_value.grid(row=row_num + 3, column=4)'''

    def crypto_particle_board(self):
        # Todo particle board code
        raise_frame(Screen2)


#PopUp Screen for API and Secret Key - Crypto
class popUp_CryptoPortfolio(Toplevel):
    def __init__(self, original):
        Toplevel.__init__(self)
        self.geometry("450x300+350+230")
        self.title("SignIn Crypto Portfolio")
        label = Label(self, text=" SignIn ", font = "Arial 10 bold", height = 5)
        label.grid(row=0, column=3)
        label = Label(self, text=" Enter API Key : ", height = 2)
        label.grid(row=1, column=2)
        self.apikey = tk.StringVar(self)
        apibox = Entry(self, width = 30,textvariable=self.apikey, borderwidth = 3).grid(row=1, column=3)
        label = Label(self, text=" Enter Secret key : ", height = 2)
        label.grid(row=2, column=2)
        self.secretkey = tk.StringVar(self)
        secretbox = Entry(self,textvariable=self.secretkey,show='*', width = 30, borderwidth = 3).grid(row=2, column=3)
        btn = Button(self, text="Submit", command=lambda: self.on_close())
        btn.grid(row=4, column=3)
    def on_close(self):
        global crypto_apikey
        global crypto_secretekey
        global crypto_exchange
        global user_email
        crypto_apikey = self.apikey.get()
        crypto_secretekey = self.secretkey.get()
        print("--------------------------------")
        print(crypto_apikey)
        print(crypto_secretekey)
        print(crypto_exchange)
        print("--------------------------------")
        crypto_balance, crypto_auth = cryptocurrency.get_account_balance(
            cryptocurrency.get_exchange_key_dict(crypto_apikey, crypto_secretekey, crypto_exchange), crypto_exchange)
        if crypto_auth is not '':
            self.grab_set()
            tk.messagebox.showwarning("Error", crypto_auth)
            self.apikey.set("")
            self.secretkey.set("")

        else:
            NCTdatabase.createCcxtUserKeyCollection(user_email,
                                                    crypto_apikey, crypto_secretekey, crypto_exchange)
            self.destroy()
            root.update()
            root.deiconify()
            crypto_protfolio_page(Screen3)
            raise_frame(Screen3)


#Crypto Login Screen
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
        Label(crptoportfolio, text=" ", width=82, height = 8 ).grid(row=8, column=5, columnspan = 2)
        btn_crypto_login = Button(crptoportfolio, text="Go back", command=lambda: raise_frame(frame1),
                                  width=15)
        btn_crypto_login.grid(row=10, column=6, sticky="se")

        # on change dropdown value
        def change_dropdown(*args):
            global crypto_exchange
            crypto_exchange = tkvar.get()
            print(tkvar.get())
        # link function to change dropdown
        tkvar.trace('w', change_dropdown)
    def signUp(self):
        popUp_CryptoPortfolio(self)


# Forex Login Screen
class forex_portfolio_login_screen():

    my_trader: Fx.Forex_trader = None
    username = ""
    password = ""
    auth_token = ""

    def __init__(self,forex_portfolio):

        self.forex_username = StringVar()
        label = Label(forex_portfolio, text="          LOGIN ", font = "Arial 10 bold", width = 50, height = 7)\
            .grid(row=1, column=2, columnspan = 2, sticky= "w")
        blank = Label(forex_portfolio,text=" ", width=40).grid(row = 2, column = 1)
        label = Label(forex_portfolio, text=" Enter Username : ", width = 30, height = 2)
        label.grid(row=3, column=2, sticky="e")

        self.uname = tk.StringVar()
        forex_username = Entry(forex_portfolio,textvariable=self.uname, width = 30, borderwidth = 3)\
            .grid(row=3, column=3, sticky = "w")
        label = Label(forex_portfolio, text=" Enter Password :  ", width = 30, height = 2)
        label.grid(row=4, column=2, sticky="e")

        self.pwd = tk.StringVar()
        forex_password = Entry(forex_portfolio,textvariable=self.pwd,show='*', width = 30, borderwidth = 3)\
            .grid(row=4, column=3, sticky = "w")
        # reg = forex_portfolio.register(self.validate_input)
        label = Label(forex_portfolio, text=" Enter AppKey :    ", width = 30, height = 2)
        label.grid(row=5, column=2, sticky="e")

        self.appkey = tk.StringVar()
        forex_appkey = Entry(forex_portfolio,textvariable=self.appkey, width = 30, borderwidth = 3)\
            .grid(row=5, column=3, sticky = "w")
        # apibox.config(validate="key",validatecommand= (reg,'%P'))
        btn_submit = Button(forex_portfolio, text="Submit",command=lambda:self.validate_input(self.uname.get(),self.pwd.get(), self.appkey.get()))\
            .grid(row=6, column=3, sticky="w")
        Label(forex_portfolio, text=" ", width=70, height=4).grid(row=7, column=3)
        btn_forex_login = Button(forex_portfolio, text="Go back", command=lambda: raise_frame(frame1), width=15)
        btn_forex_login.grid(row=8, column=4, sticky="s")

    def validate_input(self, uname, pwd, appkey):
        print("SELF Validate Input")
        # validate for blank strings in any of the fields before API call
        if uname == "" or pwd == "" or appkey == "":
            tk.messagebox.showwarning("Error", "All fields are required!")

        elif forex_portfolio_login_screen.my_trader is None:  # added elif to if
            # Login to Forex account with given uname, pwd
            forex_portfolio_login_screen.my_trader = Fx.Forex_trader(uname, pwd, appkey)

            if forex_portfolio_login_screen.my_trader.login():
                print("Login successful!/n")
                forex_portfolio_login_screen.username = uname
                print("Session id = ", forex_portfolio_login_screen.my_trader.auth_token)
                # Goto portfolio page
                forex_portfolio_page(Screen4, uname, forex_portfolio_login_screen.my_trader.auth_token)

                # raise_frame(Screen4)
            else:
                forex_portfolio_login_screen.my_trader = None
                print("Unable to login!")
                tk.messagebox.showwarning("Error", "Username/Password do not match!")
                # Set the Password input box to blank
                # self.uname.set("")
                self.pwd.set("")
                # raise_frame(frame1)
        else:
            # Goto portfolio page
            forex_portfolio_page(Screen4, uname, forex_portfolio_login_screen.my_trader.auth_token)
            # raise_frame(Screen4)


# RobinHood login Screen
class robinhood_login_screen():
    def __init__(self,robinhood_portfolio):
        label = Label(robinhood_portfolio, text="          LOGIN ", font = "Arial 10 bold", width = 50, height = 7)\
            .grid(row=1, column=2, columnspan = 2, sticky= "w")
        blank = Label(robinhood_portfolio, text=" ", width=40).grid(row = 2, column = 1)
        label = Label(robinhood_portfolio, text=" Enter Username : ", width = 30, height = 2)
        label.grid(row=3, column=2, sticky="e")
        robin_name_box = Entry(robinhood_portfolio, width = 30, borderwidth = 3)\
            .grid(row=3, column=3, sticky = "w")
        label = Label(robinhood_portfolio, text=" Enter Password : ", width = 30, height = 2)
        label.grid(row=4, column=2, sticky="e")
        robin_password_box = Entry(robinhood_portfolio,show='*', width = 30, borderwidth = 3)\
            .grid(row=4, column=3, sticky = "w")
        btn = Button(robinhood_portfolio, text="Submit",command=lambda :self.signUp_token_robinhood())\
            .grid(row=5, column=3, sticky = "w")
        Label(robinhood_portfolio, text = " ", width = 70, height = 6 ).grid(row = 6, column = 3)
        btn_robinhood_login = Button(robinhood_portfolio, text="Go back", command=lambda: raise_frame(frame1), width=15)
        btn_robinhood_login.grid(row=7, column=4, sticky="s")
    def signUp_token_robinhood(self):
        popUp_token_robinhoodPortfolio(self)


#RobinHood Token Popup
class popUp_token_robinhoodPortfolio(Toplevel):
    def __init__(self, original):
        self.original_frame = original
        Toplevel.__init__(self)
        self.transient(Screen2)
        self.geometry("350x150+350+230")
        self.title("Enter Token for Authentication")
        self.lift()
        Label(self, text= "  ", height = 2).grid(row=0, column=1)
        label = Label(self, text="     Enter Token : ", height = 2)
        label.grid(row=1, column=2)
        token_robinhood_box = Entry(self, borderwidth = 3).grid(row=1, column=3)
        btn_robinhood = Button(self, text="Submit", command=lambda: self.on_close())
        btn_robinhood.grid(row=4, column=3)
    def on_close(self):
        self.destroy()
        root.update()
        root.deiconify()
        raise_frame(Screen5)


#RobinHood Protfolio page
class robinhood_portfolio_page():
    def __init__(self,Screen5):
        header_Current_balance = Label(Screen4, text="Current Balance", bg="black",
                                       fg="white", font="Times 9 bold", width=12)
        header_Current_balance.grid(row=3, column=0)
        header_Current_blank = Label(Screen4, text="    ", bg="black",
                                     fg="white", font="Times 9 bold", width=20)
        header_Current_blank.grid(row=3, column=1)

        header_currency = Label(Screen4, text="Currency   ", bg="black",
                                fg="white", font="Times 9 bold", width=12)
        header_currency.grid(row=3, column=2, sticky="e")
        header_blank = Label(Screen4, text="   ", bg="black",
                             fg="white", font="Times 9 bold", width=20)
        header_blank.grid(row=3, column=3, sticky="news")
        header_date = Label(Screen4, text="Date", bg="black",
                            fg="white", font="Times 9 bold", width=30)
        header_date.grid(row=3, column=4, sticky="news")
        #btn_Robinhood_portfolio = Button(Screen5, text="Go back", command=lambda: raise_frame(Screen2), width=20)
        #btn_Robinhood_portfolio.grid(row=15, column=1, sticky="s")
        btn_Robinhood_portfolio = Button(Screen5, text="Activate/Deactivate AI Access Hardware Key",
                                      command=lambda: self.robin_particle_board() ,width=40)
        btn_Robinhood_portfolio.grid(row=2, column=3, columnspan= 4,  sticky="e")

    def robin_particle_board(self):
        # Todo particle board code
        raise_frame(Screen2)


# Forex Portfolio page
class forex_portfolio_page():

    my_trader: Fx.Forex_trader

    def __init__(self, Screen4, username, session):

        self.my_trader = forex_portfolio_login_screen.my_trader
        raise_frame(Screen4)

        header_Current_balance = Label(Screen4, text="Current Balance", bg="black",
                                       fg= "white", font="Times 9 bold", width=12)
        header_Current_balance.grid(row=3, column=0)
        header_Current_blank = Label(Screen4, text="    ", bg="black",
                                       fg="white", font="Times 9 bold", width=20)
        header_Current_blank.grid(row=3, column=1)

        header_currency = Label(Screen4, text="Currency   ", bg="black",
                                fg= "white", font="Times 9 bold", width=12)
        header_currency.grid(row=3, column=2, sticky= "e")
        header_blank = Label(Screen4, text="   ", bg="black",
                                fg="white", font="Times 9 bold", width=20)
        header_blank.grid(row=3, column=3, sticky="news")
        header_date = Label(Screen4, text="Date", bg="black",
                            fg= "white", font="Times 9 bold", width=30)
        header_date.grid(row=3, column=4, sticky="news")
        btn_forex_portfolio = Button(Screen4, text="Activate/Deactivate AI Access Hardware Key",
                                             command=lambda: self.forex_particle_board() ,width=40)
        btn_forex_portfolio.grid(row=2, column=3, columnspan= 4,  sticky="e")

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
        # insertedId = dm.create_forex_db(self.my_trader.username, result['Cash'], result['Currency'])
        insertedId = NCTdatabase.create_forex_db(self.my_trader.username, result['Cash'], result['Currency'])

        # Fetch all previous records from the database and show on the screen
        n, result = NCTdatabase.fetch_forex_data(self.my_trader.username)

        for rownum in range(n):
            current_balance_value = Label(Screen4, text=result[rownum]["balance"], bg= "black", fg = "white")
            current_balance_value.grid(row=rownum + 4, column=0, sticky = "e")
            currency_value = Label(Screen4, text=result[rownum]["currency"], bg= "black", fg = "white")
            currency_value.grid(row=rownum + 4, column=2, sticky = "e")
            date_value = Label(Screen4, text=result[rownum]["date"], bg= "black", fg = "white")
            date_value.grid(row=rownum + 4, column=4)

    def forex_particle_board(self):
        # Todo particle board code
        raise_frame(Screen2)

#Finance home page
def financeHome(financehome):
    header_id = Label(financehome, text="ID", bg="light grey", font="Times 9 bold", width=20)
    header_id.grid(row=2, column=0, sticky="news")
    header_name = Label(financehome, text="NAME", bg="white", font="Times 9 bold", width=20)
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


#Main Class
class MainApp():

    # Define the Handler for Delete Window protocol
    root.protocol("WM_DELETE_WINDOW", on_closing)

    Label(frame1, text = " ", bg = "black", width = 25).grid(row = 0, column = 0)
    img = PhotoImage(file="nct_new.png")
    image1 = tk.Label(frame1, image=img, width=1330 , height = 220, bg = "black").\
        grid(row=0, column=0, columnspan = 2)
    label1 = tk.Label(frame1, text="Welcome to NCT Software", font="Arial 15 bold", fg="light yellow",
                      height=2, width=50, bg ="black"). \
        grid(row=1, column=0, columnspan = 2)

    style = ttk.Style()
    ttk.Style().configure("TNotebook.Tab", bordercolor="black")
    nb = ttk.Notebook(frame1, width=1130, height=450)
    nb.grid(row=2, column=0, columnspan=10, sticky = "w")

    myportfolio = ttk.Frame(nb)

    #style.configure(myportfolio, background='black')
    nb.add(myportfolio, text='    My Portfolio                                                       '
                             '                                                                      ')

    # LoginFrame(myportfolio)
    MyPortfolio(myportfolio)
    financehome = ttk.Frame(nb)
    nb.add(financehome, text='    Finance Home                                                         '
                            '                                                                      ')
    financeHome(financehome)
    style.configure('TNotebook.Tab', font = "Arial 10 bold")
    # Screen 2
    img_screen2 = PhotoImage(file="nct_screen2.png")
    image_Screen2 = tk.Label(Screen2, image=img_screen2, width=1000, height=150, bg="black"). \
        grid(row=0, column=0, columnspan = 1, sticky="e")
    Label(Screen2, text="My Portfolios", font="Arial 12 bold", width=110, bg="black", fg="white", height=2) \
        .grid(row=1, column=0, columnspan = 4, sticky="w")

    #blank_screen2 = Label(Screen2, text=" ", width=5).grid(row=2, column=0)
    nb2 = ttk.Notebook(Screen2, width=1500, height=500)
    nb2.grid(row=5, column=0, columnspan=3)

    # tab1 - Crypto Login page
    cryptoportfolio = ttk.Frame(nb2)
    nb2.add(cryptoportfolio, text='    Crypto Portfolio                                                             ')
    crypto_login_screen(cryptoportfolio)
    # Screen 3 Crypto Portfolio
    img_screen3 = PhotoImage(file="nct_screen2.png")
    image_Screen3 = tk.Label(Screen3, image=img_screen3, width=180, height=150, bg="black"). \
        grid(row=0, column=1, columnspan=4)
    Label(Screen3, text="Crypto Portfolio", font="Arial 12 bold", width=90, bg="black", fg="white", height=2) \
        .grid(row=1, column=1, columnspan=4, sticky="e")
    Button(Screen3, text="Go Back", width=15, height=1, command=lambda: raise_frame(Screen2)).\
        grid(row=0, column=3,sticky = "ne" )
    #Label(Screen3, text="Crypto Portfolio", font="Arial 10 bold ", width=140, bg="rosybrown").grid(row=0, column=0,columnspan=6)
    #Display Crypto Portfolio page
    crypto_protfolio_page(Screen3)

    # tab2 - Forex Login page
    forex_portfolio = ttk.Frame(nb2)
    nb2.add(forex_portfolio, text='    Forex Portfolio                                                              ')
    forex_portfolio_login_screen(forex_portfolio)
    #Screen #4 - Forex Portfolio
    img_screen4 = PhotoImage(file="nct_screen2.png")
    image_Screen4 = tk.Label(Screen4, image=img_screen4, width=180, height=150, bg="black"). \
        grid(row=0, column=1, columnspan=4)
    Label(Screen4, text="Forex Portfolio", font="Arial 12 bold", width=90, bg="black", fg="white", height=2) \
        .grid(row=1, column=1, columnspan=4, sticky = "e")
    Button(Screen4, text="Go Back", width=15, height=1, command=lambda: clean_up(Screen2)).\
        grid(row=0, column=5, sticky="ne")
    #Label(Screen4, text="Forex Portfolio", font="Arial 10 bold ", width=140, bg="rosybrown").grid(row=0, column=0,
    #                                                                                               columnspan=6)

    # forex_portfolio_page(Screen4)

    # tab3 - Robinhood login page
    robinhood_portfolio = ttk.Frame(nb2)
    nb2.add(robinhood_portfolio, text='    Robinhood Portfolio                                                      ')
    robinhood_login_screen(robinhood_portfolio)
    #Screen 5
    img_screen5 = PhotoImage(file="nct_screen2.png")
    image_Screen5 = tk.Label(Screen5, image=img_screen5, width=180, height=150, bg="black"). \
        grid(row=0, column=1, columnspan=4)
    Label(Screen5, text="RobinHood Portfolio", font="Arial 12 bold", width=90, bg="black", fg="white", height=2) \
        .grid(row=1, column=1, columnspan=4, sticky="e")
    Button(Screen5, text="Go Back", width=15, height=1, command=lambda: raise_frame(Screen2)).grid(row=0, column=5,
                                                                                                   sticky="ne")
    #Label(Screen5, text="Robinhood Portfolio", font="Arial 10 bold ", width=140, bg="rosybrown").grid(row=0, column=0,
    #                                                                                               columnspan=6)
    robinhood_portfolio_page(Screen5)


if __name__ == "__main__":
    raise_frame(frame1)
    MainApp()
    root.mainloop()
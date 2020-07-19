import pymongo
from pymongo.operations import DeleteOne
import tkinter as tk
from tkinter import messagebox
from datetime import datetime

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

# The no. of rows (balance records) to be shown on screen
MIN_RECORDS = 5

class NCTdatabase:

    @staticmethod
    def createDatabase(signupname, signupemail, signuppassword, signupDOB):
        error_msg = " "
        mydb = myclient["NCTSoftware"]
        mytable = mydb["Signup"]
        for x in mytable.find():
            print(x)
            current_email = x['email_id']
            if (signupemail == current_email):
                error_msg = "Error"
                print("Inside if", x)
            # print("errormsg", error_signup)
            return error_msg

        if (error_msg == " "):
            # print("Inside else", x)
            emp_rec = {"name": signupname, "email_id": signupemail, "password": signuppassword,
                       "DOB": signupDOB}
            mytable.insert_one(emp_rec)
            tk.messagebox.showinfo("Message", "Sucessfully signed up")
            return

    @staticmethod
    def validateUser(signupemail, signuppassword):
        error_msg = "Error"
        mydb = myclient["NCTSoftware"]
        mytable = mydb["Signup"]
        for x in mytable.find():
            print(x)
            current_email = x['email_id']
            current_password = x['password']
            if (signupemail == current_email and signuppassword == current_password):
                error_msg = " "
        return error_msg

    @staticmethod
    def checkIfCcxtKeyExistsInDB(useremail, exchagename):
        '''
        This Method check is the API Key and Secret Key for the user have already been saved
        :param useremail    email Id of the user
        :param exchagename:  exchange to which the user whats to login in like binance, livecoin etc
        :return:
        '''
        mydb = myclient["NCTSoftware"]
        mycol = mydb["CcxtUserKeys"]
        for x in mycol.find():
            user_email = x['UserEmail']
            exchage_name = x['UserExchName']
            if useremail == user_email and exchagename == exchage_name:
                return True
            else:
                return False

    @staticmethod
    def createCcxtUserKeyCollection(useremail, apikey, skey, exchangename):
        '''
        This method creates the document in the collection for saving useremail, exchange and its respective keys, if
        it does not exists or updates a existing one if it exists
        :param apikey:
        :param skey:
        :param exchagename:
        :return:
        '''
        try:
            mydb = myclient["NCTSoftware"]
            mycol = mydb["CcxtUserKeys"]
            mycol.find_one_and_update({"UserEmail": useremail, "UserExchName": exchangename},
                                      {"$set": {"UserEmail": useremail, "API_key": apikey, "Secret_key": skey,
                                                "UserExchName": exchangename}},
                                      upsert=True)
            print("Apikey and Secret key added to the collection")
        except Exception as e:
            print("Collection not updated: ", str(e))

    @staticmethod
    def saveCurrentBalanceCcxt(useremail, exchangename, currentbalace):
        '''
        This Method saves the user's current balance at the time of every login
        :param exchangename: Exchange name like Binance, livecoin etc
        :param currentbalace: Total balance at the time of login
        :return:
        '''
        mydb = myclient["NCTSoftware"]
        mycol = mydb["CcxtUserBalances"]
        mycol.insert_one({"UserEmail": useremail, "UserExchName": exchangename, "CurrentBalance": currentbalace,
                          "date": datetime.today().strftime("%m/%d/%Y, %H:%M:%S")})
        if mycol.count_documents({"UserEmail": useremail}) > 5:
            mycol.delete_one({"UserEmail": useremail})
            print("Document deleted")
        else:
            print(mycol.count())

    @staticmethod
    def fetchCcxtCurrentBalnces(useremail, exchangename):
        '''
        This Method fetches the users current balances from the collection
        :param exchangename: Exchange name like Binance, livecoin etc
        :return: List of documents or records of the balances
        '''
        mydb = myclient["NCTSoftware"]
        mycol = mydb["CcxtUserBalances"]
        mylist = []
        for x in mycol.find({"UserEmail": useremail, "UserExchName": exchangename}):
            mylist.append(x)
            print(x)
        return mylist

    @staticmethod
    def create_forex_db(name, balance, currency):
        '''
        Creates a database for storing all Forex user's balance info, if it doesn't exist
        :param name: usename
        :param balance: account balance of the user
        :param currency: currency
        :return: inserted document's id
        '''
        mydb = myclient["forexdatabase"]
        # Check if the "customers" collection exists
        collist = mydb.list_collection_names()
        if "customers" in collist:
            print("Collection exists!")

        # Create a table/collection called customer-<username> for this user
        dbname = "customer_" + name
        mycollection = mydb[dbname]
        # Add a record/document to the collection
        mydict = {"name": name,
                  "balance": balance,
                  "currency": currency,
                  "date": datetime.today().strftime("%m/%d/%Y, %H:%M:%S")}

        x = mycollection.insert_one(mydict)
        return x.inserted_id

    @staticmethod
    def fetch_forex_data(username):
        '''
        Connects to Username's collection and returns all documents/records
        :param username: The usename whose records need to be pulled
        :return: no. of documents and a list with all documents
        '''
        # print("Fetch-forex-data: username = ", username)
        mydb = myclient["forexdatabase"]

        mycollection = mydb["customer_" + username]
        # mycollection.find().count()
        mylist = []
        # Using find() method for Mongodb to select all documents/records
        for x in mycollection.find():
            mylist.append(x)
            # print(x)
        return mycollection.find().count(), mylist

    @staticmethod
    def update_collection(username):
        '''
        Connects to Username's collection and deletes first n rows so that the Collection
        contains only 6 documents
        :param username: The username who is logged in
        :return:
        '''
        mydb = myclient["forexdatabase"]
        mycollection = mydb["customer_" + username]
        n = mycollection.count_documents({})
        # print("Initial count= ", n)
        if n > MIN_RECORDS:
            result = mycollection.bulk_write([DeleteOne({})] * (n - MIN_RECORDS))
            # print(result.deleted_count)
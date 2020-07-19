import pymongo
from pymongo.operations import DeleteOne
from datetime import datetime

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

# The no. of rows (balance records) to be shown on screen
MIN_RECORDS = 5

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
    #mycollection = mydb["customers"]
    mycollection = mydb[dbname]
    # Add a record/document to the collection
    mydict = {"name": name,
              "balance": balance,
              "currency": currency,
              "date": datetime.today().strftime("%m/%d/%Y, %H:%M:%S")}

    x = mycollection.insert_one(mydict)
    print("Inserted Id: ",x.inserted_id)

    #update_collection(name)
    return x.inserted_id

def fetch_forex_data(username):
    '''
    Connects to Username's collection and returns all documents/records
    :param username: The usename whose records need to be pulled
    :return: no. of documents and a list with all documents
    '''
    #print("Fetch-forex-data: username = ", username)
    mydb = myclient["forexdatabase"]

    mycollection = mydb["customer_"+username]
    #mycollection.find().count()
    mylist = []
    # Using find() method for Mongodb to select all documents/records
    for x in mycollection.find():
        mylist.append(x)
        # print(x)
    return mycollection.find().count(), mylist

def update_collection(username):
    '''
    Connects to Username's collection and deletes first n rows so that the Collection
    contains only 6 documents
    :param username: The username who is logged in
    :return:
    '''
    mydb = myclient["forexdatabase"]
    mycollection = mydb["customer_"+username]
    n = mycollection.count_documents({})
    #print("Initial count= ", n)
    if n > MIN_RECORDS:
        result = mycollection.bulk_write([DeleteOne({})] * (n - MIN_RECORDS))
        #print(result.deleted_count)


if __name__ == '__main__':
    update_collection("DA545354")
import pymongo
import datetime

client = pymongo.MongoClient('localhost', 27017)

def createLoginDatabase(em, pas, name, dob):
    error = ''
    db = client["NCTApp"]
    col = db["loginCol"]

    query = {"Email": em}

    result = col.find_one(query)

    ## User already exists in collection
    if not result is None:
        error = 'Error'
        return error

    ## User doesn't exist so add to collection
    sign_up_info = {"Email": em,
                    "Password": pas,
                    "UName": name,
                    "DOB": dob
                    }

    col.insert_one(sign_up_info)
    return

def validateUser(em, passw):
    message = ''
    db = client["NCTApp"]
    col = db["loginCol"]

    query = {"Email": em, "Password": passw}

    result = col.find_one(query)

    if result is None:
        ## User doesn't exist or not found
        message = 'Invalid credentials'
    else:
        ## User found
        message = 'Success'

    return message

def createCcxtUserKeyCol(email, apikey, skey, exname):
    try:
        db = client["NCTApp"]
        col = db["ccxtUserKeys"]
        col.find_one_and_update({"UserEmail": email, "UserExchName": exname},
                                      {"$set": {"UserEmail": email, "API_key": apikey, "Secret_key": skey,
                                                "UserExchName": exname}},
                                      upsert=True)

    except Exception as e:
        print(str(e))

def ccxtKeyCheck(email, exname):
    message = ''
    db = client["NCTApp"]
    col = db["ccxtUserkeys"]
    query = {"UserExchName": exname}
    result = col.find_one(query)

    if result is None:
        message = "User not found"
    else:
        message = "User exists"

    return message

def saveCurrentCcxtBalance(email, exname, balance):
    db = client['NCTApp']
    col = db['ccxtUserBalances']
    col.insert_one({"UserEmail": email, "UserExchName": exname, "CurrentBalance": balance,
                          "date": datetime.today().strftime("%m/%d/%Y, %H:%M:%S")})

def getCcxtBalance(email, exname):
    db = client['NCTApp']
    col = db['ccxtUserBalances']

    query = {"UserEmail": email, 'UserExchName': exname}
    arr = []
    for x in col.find_one(query):
        arr.append(x)
    return arr

def createForexDatabase(name, balance, curr):
    db = client['forexData']

    dbname = "customer_" + name
    col = db[dbname]

    doc = {"Name": name,
            "Balance": balance,
            "Currency": curr,
            "Date": datetime.today().strftime("%m/%d/%Y, %H:%M:%S")}

    result = col.insert_one(doc)

    return result.inserted_id

def fetchForexData(name):
    db = client["forexData"]
    dbname = "customer_" + name
    col = db[dbname]

    query = {"Name": name}
    arr = []

    if col.find_one(query) is None:
        return "User not found"
    for x in col.find_one(query):
        arr.append(x)
    return arr

NCT Trading Software

The program starts by running the file NCT_gui.py from NCT folder.

1. An NCT sotware login screen shows up.
2. Create an acccount and login to NCT software.
3. Select the portfolio : Robinhood/Crypto currency/ Forex.
4. Enter login credentials for the selected portfolio.
5. Portfolio screen pops up with information of your last 5 account balances and a AI Hardware Access key.
6. MongoDB database is used for storing login information and user's previous portfolio account balances.


To login to all three portfolios Username/Password are required

AI Hardware Access Key:
1. This button is given for enabling users to get Buy/Sell signals from the Software on the Server and also to perform actual Buy/Sell orders on behalf of the user.
2. This functionality is yet to be implemented. The files particleBoardServer.py and particleBoardClient.py are in the repo which can be used as reference for accessing the particle devices like Boron, Argon.


The files NCT_GUI2.py is the older gui file which shows Account Balance records of user on screen without using any database.

<<<<<<< HEAD
import six

if six.PY3:
    from Robinhood.Robinhood import Robinhood
    my_trader = Robinhood()
    #logged_in = my_trader.login(username="nn131983", password="Hittingstride*13")
    #stock_instrument = my_trader.instruments("GEVO")[0]
    quote_info = my_trader.quote_data("GEVO")
    print(quote_info)
else:
    from Robinhood import Robinhood
    import exceptions as RH_exception
=======
import six

if six.PY3:
    from Robinhood.Robinhood import Robinhood
    my_trader = Robinhood()
    #logged_in = my_trader.login(username="nn131983", password="Hittingstride*13")
    #stock_instrument = my_trader.instruments("GEVO")[0]
    quote_info = my_trader.quote_data("GEVO")
    print(quote_info)
else:
    from Robinhood import Robinhood
    import exceptions as RH_exception
>>>>>>> added triangular arbitrage folder

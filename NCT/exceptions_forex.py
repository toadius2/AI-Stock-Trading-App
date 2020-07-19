
class ForexException(Exception):
    """
        Wrapper for custom Robinhood library exceptions
    """

    pass

class LoginFailed(ForexException):
    """
        Unable to login to Robinhood

    """
    print("Unable to login")
    pass
import requests

base_url = 'https://newapi.bilaxy.com'

class Bilaxy:
    def __init__(self):
        pass

    def currencies(self):
        url = base_url + '/v1/currencies'
        return requests.get(url).json()


if __name__ == '__main__':
    bilaxy = Bilaxy()
    print(bilaxy.currencies().keys())

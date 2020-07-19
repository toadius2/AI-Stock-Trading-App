from square.client import Client
import uuid
import webbrowser

# prod_access_token = 'EAAAEKaqNdNBBRU1JWpKOUCltfjyhK85YSUR4NonoJjX2S9S-gIt2Ms-JzjZASO3'
# prod_app_id = 'sq0idp-cBQ1GkD7YpvD9ZoUz16M5A'
# prod_location_id = '4H2EQKT97R4PF'
#
# sandbox_access_token = 'EAAAEJ2GCY1bPcE29YfE_2IBPWbRplBn4v1MLH2HwCIbUB02hSEqucHTo8uNRXFp'
# sandbox_app_id = 'sandbox-sq0idb-d3KfTMmS1k64HNBQOFNhRg'
# sandbox_location_id = 'F0J53AJW9S8JC'

class Square:
    def __init__(self):
        self.client = Client(
            access_token = 'EAAAEKaqNdNBBRU1JWpKOUCltfjyhK85YSUR4NonoJjX2S9S-gIt2Ms-JzjZASO3',
            environment = 'production'
        )
        self.idempotency_key = str(uuid.uuid1())
        self.sandbox_location_id = 'F0J53AJW9S8JC'
        self.prod_location_id = '4H2EQKT97R4PF'
        self.dicount_codes = ['rcvi13', 'aiv13', 'v13er', 'v13ai', 'aivision', 'vision3ai', 'visionai', 'ai2019', 'dsho', 'vthirteenai',
            'V13Ai', 'Thirteenvision', 'Alphavision13', '2019ai', '5ai', 'AIVision13', 'xavai', 'ngthon13', 'songAI', 'prust13ai', 'kyle13', 'wolfAI',
            'CottAI13', 'MinAI', 'Sami13', 'PradVision13']

    def transactions(self):
        transactions_api = self.client.transactions
        result = transactions_api.list_transactions(self.prod_location_id)
        if result.is_success():
            print(result.body)
        else:
            print(result.errors)

    def retrieve_order(self, order_id):
        orders_api = self.client.orders
        body={}
        body['order_ids'] = [order_id]
        result = orders_api.batch_retrieve_orders(self.prod_location_id, body)
        if result.is_success():
            print(result.body)
        else:
            print(result.errors)
        return result.body

    # Generate url for checkout link
    def create_url(self, apply_discount=False):

        body = {}
        body['idempotency_key'] = self.idempotency_key
        body['order'] = {}
        body['order']['reference_id'] = 'reference_id'
        body['order']['line_items'] = []

        body['order']['line_items'].append({})
        body['order']['line_items'][0]['name'] = 'AI Service'
        body['order']['line_items'][0]['quantity'] = '1'
        body['order']['line_items'][0]['base_price_money'] = {}
        if apply_discount:
            body['order']['line_items'][0]['base_price_money']['amount'] = 2000
            body['order']['line_items'][0]['base_price_money']['currency'] = 'USD'
        else:
            body['order']['line_items'][0]['base_price_money']['amount'] = 2500
            body['order']['line_items'][0]['base_price_money']['currency'] = 'USD'

        checkout_api = self.client.checkout
        result = checkout_api.create_checkout(self.prod_location_id, body)

        response = {}
        if result.is_success():
            response = result.body
        elif result.is_error():
            print(result.errors)

        url = response['checkout']['checkout_page_url']
        page_id = response['checkout']['id']
        order_id = response['checkout']['order']['id']

        return order_id, url

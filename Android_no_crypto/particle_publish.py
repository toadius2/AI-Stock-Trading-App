import requests
import json

class Particle_publish:
    def __init__(self):
        self.headers = {}
        self.secret = '2f7c82f8baca961fbff0e657c7edfb2ad7c9b110'
        self.client_id = 'nct-app-3115'
        self.access_token = ''
        self.refresh_token = ''

    def auth(self):
        url = 'https://api.particle.io/oauth/token'
        payload = {
            'client_id': self.client_id,
            'client_secret': self.secret,
            'grant_type': 'password',
            'password': 'Hittingstride*13',
            'username': 'noah13nelson@gmail.com'
        }
        try:
            res = requests.post(url, data=payload)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(str(e))
            return 'error'

        if 'access_token' in data:
            self.access_token = data['access_token']
            self.refresh_token = data['refresh_token']
            return 'success'
        else:
            return 'failed'

    def publish_event(self, event_name, data=''):
        url = 'https://api.particle.io/v1/devices/events'
        payload = {
            'access_token': self.access_token,
            'name': event_name,
            'private': 'true'
        }
        if data != '':
            payload['data'] = data
        try:
            res = requests.post(url, data=payload).json()
            print(res)
        except Exception as e:
            print(str(e))
            return 'error'

        if res['ok']:
            return 'success'
        else:
            return 'failed'

if __name__ == '__main__':
    part = Particle_publish()
    part.auth()
    print(part.publish_event('Event', 'exchange'))

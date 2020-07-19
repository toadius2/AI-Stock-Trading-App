import json

with open('terms.txt', 'w') as f:
    data = {'agreed': False}
    json.dump(data, f)

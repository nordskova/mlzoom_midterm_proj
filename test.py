import requests

url = 'http://127.0.0.1:63949/predict'
data = {"text": "Life is good!"}

result = requests.post(url, json=data).json()
print(result)
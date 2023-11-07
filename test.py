import requests

url = 'http://localhost:9696/predict'
data = {"text": "Life is good!"}

result = requests.post(url, json=data).json()
print(result)
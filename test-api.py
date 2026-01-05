import requests

url = "http://localhost:9696/predict"

filename = "test-image.jpg"

with open(filename, "rb") as f:
    files = {"file": (filename, f, "image/jpeg")}
    response = requests.post(url, files=files)

print(response.json())

import requests

url = 'http://127.0.0.1:5000/classify'

file_path = 'path_to_profile_picture.jpg'  # Change to the path of the profile picture

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)
    print(response.json())

import requests
import threading
import time
import uvicorn

def start_server():
    uvicorn.run('api.main:app', host='127.0.0.1', port=8000, reload=False)

t = threading.Thread(target=start_server)
t.daemon = True
t.start()
time.sleep(3)  # Wait for server to start

base_url = 'http://localhost:8000'

# Test /api/predict for diabetes
print("Testing /api/predict for diabetes...")
response = requests.post(f'{base_url}/api/predict', json={'model_type': 'diabetes', 'features': [6, 148, 72, 35, 0, 33.6, 0.627, 50]})
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")

# Test /api/predict for heart
print("\nTesting /api/predict for heart...")
response = requests.post(f'{base_url}/api/predict', json={'model_type': 'heart', 'features': [55, 1, 170, 80, 120, 80, 1, 1, 0, 0, 1]})
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")

# Test /api/predict for stroke
print("\nTesting /api/predict for stroke...")
response = requests.post(f'{base_url}/api/predict', json={'model_type': 'stroke', 'features': [50, 0, 0, 100, 25, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]})
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")

# Test /api/predict/cardio
print("\nTesting /api/predict/cardio...")
response = requests.post(f'{base_url}/api/predict/cardio', json={'features': [55, 1, 170, 80, 120, 80, 1, 1, 0, 0, 1]})
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")

# Test /api/upload (assuming a test file exists)
print("\nTesting /api/upload...")
with open('sample_report.txt', 'rb') as f:
    files = {'file': f}
    response = requests.post(f'{base_url}/api/upload', files=files)
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")

# Edge cases
print("\nTesting edge cases...")

# Invalid model type
print("Invalid model type...")
response = requests.post(f'{base_url}/api/predict', json={'model_type': 'invalid', 'features': [1,2,3]})
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")

# Wrong number of features for diabetes
print("Wrong number of features for diabetes...")
response = requests.post(f'{base_url}/api/predict', json={'model_type': 'diabetes', 'features': [1,2,3]})
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")

# Wrong number of features for heart
print("Wrong number of features for heart...")
response = requests.post(f'{base_url}/api/predict', json={'model_type': 'heart', 'features': [1,2,3,4,5,6,7,8]})
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")

# Wrong number of features for stroke
print("Wrong number of features for stroke...")
response = requests.post(f'{base_url}/api/predict', json={'model_type': 'stroke', 'features': [1,2,3,4,5]})
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")

print("\nAll API tests completed.")

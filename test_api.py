import uvicorn
import threading
import time
import requests

def start_server():
    uvicorn.run('api.main:app', host='127.0.0.1', port=8000, reload=False)

t = threading.Thread(target=start_server)
t.daemon = True
t.start()
time.sleep(2)

response = requests.post('http://localhost:8000/api/predict', json={'model_type': 'stroke', 'features': [67, 0, 1, 228.69, 36.6, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]})
print('Status:', response.status_code)
print('Response text:', response.text)

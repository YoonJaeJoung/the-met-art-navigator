import requests
import time

for _ in range(10):
    try:
        requests.get("http://localhost:8000/status")
        break
    except:
        time.sleep(1)

requests.post("http://localhost:8000/train", json={"epochs": 1, "batch_size": 256, "lr": 1e-3})

for _ in range(10):
    status = requests.get("http://localhost:8000/status").json()
    if status["training_status"] == "complete":
        break
    time.sleep(1)

res = requests.post("http://localhost:8000/search/text", json={"query": "impressionist landscape", "top_k": 3})
print("Search status:", res.status_code)
print("Search body:", res.text)

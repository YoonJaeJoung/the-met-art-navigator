import requests
import time
print("Starting train...")
r = requests.post("http://localhost:8000/train", json={"epochs": 1, "batch_size": 256, "lr": 1e-3})
print(r.json())
for _ in range(20):
    time.sleep(1)
    status = requests.get("http://localhost:8000/status").json()
    print(status)
    if status["training_status"] in ["complete", "error"]:
        break

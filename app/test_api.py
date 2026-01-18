import requests
URL = "http://127.0.0.1:8000/predict"
emails=["Congratulations! You won $1000!",
    "Hi John, can we reschedule our meeting?",
    "This is not spam, just checking in.",
    "Get cheap meds now!!!"]
for email in emails:
    payload={"email":email}
    response=requests.post(URL,json=payload)
    print(f"Email: {email}")
    print("Prediction:", response.json())
    print("-"*40)
#Automated: No manual POST requests.

#Regression-proof: If you change your model, run tests → catch errors.
import requests
import pytest
BASE_URL="http://127.0.0.1:8000"
# Test the POST /predict endpoint
def test_predict_spam():
    payload  = {"email": "Congratulations! You won $1000!"}
    response=requests.post(f"{BASE_URL}/predict",json=payload)
    assert response.status_code==200
    data=response.json()
    assert "label" in data
    assert "confidence" in data #assert checks that a condition is True, and raises an error if it’s False — used in tests to automatically verify expected behavior.
    assert data["label"] == "spam"


def test_predict_ham():
    payload={"email":"Hi John, can we reschedule our meeting?"}
    response= requests.post(f"{BASE_URL}/predict",json=payload)
    assert response.status_code==200
    data=response.json()
    assert "label" in data
    assert "confidence" in data
    assert data["label"] =="ham"


def test_missing_email_field():
    payload={}
    response=requests.post(f"{BASE_URL}/predict",json=payload)
    assert response.status_code == 422
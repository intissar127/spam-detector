📨 Spam/Ham Detector

A full ML pipeline project featuring spam detection, Dockerized API, dataset & model versioning, and real-time monitoring.

🔹 Project Overview

This project demonstrates an end-to-end MLOps workflow:

Data versioning: DVC tracks datasets and models.

Model training: LSTM model for spam classification (TensorFlow/Keras).

API: FastAPI serves predictions in real-time.

Containerization: Docker ensures consistent deployment.

Experiment tracking & monitoring: Prometheus + Grafana to track requests, latency, and performance.

UI: Streamlit interface to interact with the model easily.

📊 Features

Track dataset and model changes with DVC

Real-time predictions via FastAPI

Streamlit UI for testing emails

Prometheus metrics & Grafana dashboard for monitoring API requests and latency

Fully containerized environment with Docker
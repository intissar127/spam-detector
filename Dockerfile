FROM python:3.11-slim

WORKDIR /app
#Copies all files from your project into the container’s working directory.
COPY . /app
#Installs all Python dependencies listed in requirements.txt without caching to save space.
RUN pip install --no-cache-dir -r requirements.txt
#Creates a logs folder inside the container to store log files.
RUN mkdir -p logs
EXPOSE 8000
# Run the FastAPI app
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
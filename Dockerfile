FROM python:3.11-slim

WORKDIR /

# 1. On copie et installe les dépendances d'abord (pour le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. IMPORTANT : On copie tout le reste du projet (dont le dossier /app) COPY . .. Sans cette ligne, le conteneur était comme une boîte vide avec Python installé, mais sans votre code source à l'intérieur. Uvicorn cherchait donc un dossier app qui n'existait pas physiquement dans l'image.
COPY . .

# 3. Création des dossiers nécessaires
RUN mkdir -p logs

EXPOSE 8000

# 4. Lancement (Uvicorn trouvera maintenant le module app.main)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
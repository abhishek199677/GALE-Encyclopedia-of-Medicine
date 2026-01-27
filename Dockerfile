# 1. Use Python 3.14 (Slim version for smaller image size)
FROM python:3.14-slim-bookworm

# 2. Set environment variables to optimize Python performance in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory
WORKDIR /app

# 4. Install essential system dependencies
# These are often needed for libraries like numpy or pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install requirements separately (faster builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your medical bot code
COPY . .

# 7. Expose the port your Flask app uses
EXPOSE 8080

# 8. Run the application
CMD ["python", "app.py"]
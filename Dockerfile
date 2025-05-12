FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run tests (can be disabled with build-arg SKIP_TESTS=true)
ARG SKIP_TESTS=false
RUN if [ "$SKIP_TESTS" = "false" ]; then python run_tests.py; fi

# Expose port
EXPOSE 8050

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]

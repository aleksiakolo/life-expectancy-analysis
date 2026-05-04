FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional but safe)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install package
RUN pip install --upgrade pip
RUN pip install -e ".[dev]"

# Default command
ENTRYPOINT ["lifeexp"]
CMD ["--help"]

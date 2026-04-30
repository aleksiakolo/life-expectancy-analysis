# 1. Base environment (Python + Linux)
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Avoid Python writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install system dependencies (needed sometimes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy project files into container
COPY pyproject.toml README.md ./
COPY life_expectancy/ life_expectancy/
COPY configs/ configs/

# 6. Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# 7. Default command to run
CMD ["lifeexp"]

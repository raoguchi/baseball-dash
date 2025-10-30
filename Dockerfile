<<<<<<< HEAD
FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
=======
# ---- base build (dependencies) ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (git for pybaseball, plus basic build tooling if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy metadata first to leverage Docker layer caching
COPY pyproject.toml README.md ./
COPY src ./src

# Install package in editable mode (or regular if you prefer)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy the Streamlit app (optional; used in default CMD)
COPY app.py ./app.py

# Create mount points for persistent cache/artifacts
RUN mkdir -p /app/data/cache /app/artifacts

# ---- runtime image ----
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /app /app

# Expose Streamlit port
EXPOSE 7860

# Default: run Streamlit app (Hugging Face Space style)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
>>>>>>> 63696d41 (Initial Commit)

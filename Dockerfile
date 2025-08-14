
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .[dev,evaluation]

# Copy source code
COPY src/ src/
COPY config/ config/
COPY tests/ tests/

# Create directories for data and results
RUN mkdir -p data results logs cache temp

# Set environment variables
ENV PYTHONPATH=/app/src
ENV TOKENIZERS_PARALLELISM=false

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["rope-eval", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import rope_long_context_evaluation_suite; print('Health check passed')" || exit 1

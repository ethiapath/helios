# Helios Trading Bot - Dockerfile
# Production-ready containerized deployment

FROM python:3.12-slim

# Set metadata
LABEL maintainer="Helios Trading Bot Project" \
      description="Systematic trading algorithm for mean-reversion and momentum strategies" \
      version="1.0"

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# Create app user for security (non-root)
RUN groupadd -r helios && \
    useradd -r -g helios -d /app -s /bin/bash helios

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/state /app/config && \
    chown -R helios:helios /app

# Copy application code
COPY --chown=helios:helios . .

# Ensure executable permissions
RUN chmod +x run.py

# Create volume mount points for persistent data
VOLUME ["/app/logs", "/app/state"]

# Switch to non-root user
USER helios

# Health check to ensure the application is running properly
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m --retries=3 \
    CMD python -c "from helios_bot.main_controller import HeliosMainController; \
                   controller = HeliosMainController(); \
                   status = controller.health_check(); \
                   exit(0 if status.get('overall_healthy', False) else 1)" || exit 1

# Expose port for potential monitoring/API endpoints (if added in future)
EXPOSE 8080

# Set the default command
CMD ["python", "run.py"]

# Build arguments for customization
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Additional metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="helios-trading-bot" \
      org.label-schema.description="Automated systematic trading algorithm" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Production deployment notes:
# 1. Build: docker build -t helios-trading-bot .
# 2. Run: docker run -d --name helios-bot --env-file .env \
#         -v $(pwd)/logs:/app/logs -v $(pwd)/state:/app/state \
#         helios-trading-bot
# 3. For production, consider using docker-compose with proper networking,
#    monitoring, and orchestration setup

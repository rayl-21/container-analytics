# Container Analytics Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Container Analytics in a production environment using Docker containers with full monitoring, logging, and alerting capabilities.

## Quick Start

```bash
# 1. Clone and prepare environment
git clone <repository-url>
cd container-analytics
cp .env.production .env

# 2. Configure environment variables (see Configuration section)
vim .env

# 3. Build and start core services
docker-compose -f deployment/docker/docker-compose.yml up -d

# 4. Start with monitoring (optional)
docker-compose -f deployment/docker/docker-compose.yml --profile monitoring up -d

# 5. Start with reverse proxy (optional)
docker-compose -f deployment/docker/docker-compose.yml --profile production up -d
```

## Architecture

### Core Services
- **Scheduler**: Downloads images from Dray Dog cameras every 10 minutes
- **Detector**: Processes images using YOLOv8 for container detection
- **Dashboard**: Streamlit web interface for analytics visualization
- **Metrics**: Prometheus metrics server for monitoring
- **Redis**: Caching layer for improved performance

### Optional Services (Profiles)
- **Monitoring Profile**: Loki (log aggregation) + Grafana (dashboards)
- **Production Profile**: NGINX reverse proxy with SSL termination

## Prerequisites

### System Requirements
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **RAM**: Minimum 8GB, Recommended 16GB+ (4GB for detector service)
- **Storage**: Minimum 100GB SSD, Recommended 500GB+ SSD
- **Network**: Stable internet connection for image downloads
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster detection

### Software Requirements
- Docker 20.10+ with Docker Compose v2
- Git for repository management
- NVIDIA Container Toolkit (if using GPU)

### Optional Requirements
- Domain name and SSL certificates (for production)
- SMTP server for email alerts
- AWS S3 bucket for automated backups

## Configuration

### Environment Variables

Copy `.env.production` to `.env` and configure:

#### Core Configuration
```bash
# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DOWNLOAD_INTERVAL_MINUTES=10
RETENTION_DAYS=30
STREAMS="in_gate"

# Port Configuration
DASHBOARD_PORT=8501
METRICS_PORT=9090
GRAFANA_PORT=3000
```

#### Security Configuration
```bash
# Generate secure passwords
DB_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)

# Alert Configuration
ALERT_EMAIL=admin@yourcompany.com
SMTP_SERVER=smtp.yourcompany.com
SMTP_USERNAME=alerts@yourcompany.com
SMTP_PASSWORD=your-smtp-password
```

#### Performance Tuning
```bash
# Resource Limits
MAX_MEMORY_MB=4096
MAX_CPU_PERCENT=80
YOLO_BATCH_SIZE=1
YOLO_CONFIDENCE_THRESHOLD=0.5
```

### Directory Structure Setup

```bash
# Create required directories
mkdir -p data/{images,models,database}
mkdir -p logs
mkdir -p deployment/docker/{nginx/ssl,grafana/dashboards}

# Set permissions
chmod 755 data logs
chmod -R 644 deployment/docker/
```

### YOLO Model Setup

```bash
# Download YOLOv8 model (if not already present)
mkdir -p data/models
cd data/models

# Download YOLOv8x model (best accuracy)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# Or use smaller/faster models:
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt  # nano (fastest)
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt  # small
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt  # medium
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt  # large

cd ../..
```

## Deployment Options

### 1. Basic Deployment (Core Services Only)

Includes: Scheduler, Detector, Dashboard, Metrics, Redis

```bash
# Start core services
docker-compose -f deployment/docker/docker-compose.yml up -d

# Check status
docker-compose -f deployment/docker/docker-compose.yml ps

# View logs
docker-compose -f deployment/docker/docker-compose.yml logs -f
```

### 2. Monitoring Deployment

Adds: Loki (logs), Grafana (dashboards)

```bash
# Start with monitoring
docker-compose -f deployment/docker/docker-compose.yml --profile monitoring up -d

# Access Grafana at http://localhost:3000
# Default login: admin / admin (change immediately)
```

### 3. Production Deployment

Adds: NGINX reverse proxy, SSL termination

```bash
# Configure SSL certificates first
cp your-cert.pem deployment/docker/nginx/ssl/cert.pem
cp your-key.pem deployment/docker/nginx/ssl/key.pem
chmod 600 deployment/docker/nginx/ssl/*

# Start production stack
docker-compose -f deployment/docker/docker-compose.yml --profile production up -d

# Access via NGINX at http://your-domain.com
```

### 4. Full Stack Deployment

All services including monitoring and production features:

```bash
docker-compose -f deployment/docker/docker-compose.yml \
  --profile monitoring \
  --profile production \
  up -d
```

## Service Management

### Starting Services

```bash
# Start specific service
docker-compose -f deployment/docker/docker-compose.yml up -d scheduler

# Start multiple services
docker-compose -f deployment/docker/docker-compose.yml up -d scheduler detector dashboard
```

### Stopping Services

```bash
# Stop all services
docker-compose -f deployment/docker/docker-compose.yml down

# Stop specific service
docker-compose -f deployment/docker/docker-compose.yml stop scheduler
```

### Updating Services

```bash
# Pull latest images
docker-compose -f deployment/docker/docker-compose.yml pull

# Rebuild and restart
docker-compose -f deployment/docker/docker-compose.yml up -d --build --force-recreate
```

### Scaling Services

```bash
# Scale detector service for higher throughput
docker-compose -f deployment/docker/docker-compose.yml up -d --scale detector=3

# Note: Only detector service supports scaling
```

## Monitoring and Health Checks

### Health Check Endpoints

- **Dashboard**: http://localhost:8501/_stcore/health
- **Metrics**: http://localhost:9090/metrics
- **Grafana**: http://localhost:3000/api/health
- **NGINX**: http://localhost/health

### Monitoring Dashboard

Access Grafana at http://localhost:3000 (monitoring profile required):

1. Login with admin credentials
2. Navigate to "Container Analytics Monitoring" dashboard
3. Monitor key metrics:
   - Service health status
   - Image download rates
   - Detection processing queue
   - System resource usage
   - Detection accuracy

### Prometheus Metrics

Key metrics available at http://localhost:9090/metrics:

- `container_analytics_service_status` - Service health (1=healthy, 0=unhealthy)
- `container_analytics_images_downloaded_total` - Total images downloaded
- `container_analytics_detections_total` - Total container detections
- `container_analytics_processing_duration_seconds` - Processing time distribution
- `container_analytics_queue_size` - Images waiting for processing

### Log Management

Logs are stored in multiple formats:

```bash
# View real-time logs
docker-compose -f deployment/docker/docker-compose.yml logs -f scheduler

# Access structured logs (inside containers)
docker exec container-analytics-scheduler ls -la /logs/
# - app.jsonl: All application logs in JSON format
# - errors.log: Error-only logs
# - performance.log: Performance metrics
# - audit.log: Audit trail
```

## Backup and Recovery

### Automated Backups

Configure automated backups in `.env`:

```bash
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
BACKUP_S3_BUCKET=your-backup-bucket
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

### Manual Backup

```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Backup database
docker exec container-analytics-scheduler \
  sqlite3 /data/database.db ".backup '/data/backup_$(date +%Y%m%d_%H%M%S).db'"

# Backup configuration
cp .env backups/$(date +%Y%m%d)/
cp -r deployment/ backups/$(date +%Y%m%d)/

# Backup images (selective)
rsync -av --include="*.jpg" --include="*.jpeg" \
  data/images/ backups/$(date +%Y%m%d)/images/
```

### Recovery

```bash
# Stop services
docker-compose -f deployment/docker/docker-compose.yml down

# Restore database
cp backup_YYYYMMDD_HHMMSS.db data/database.db

# Restore configuration
cp backup/.env .env

# Restart services
docker-compose -f deployment/docker/docker-compose.yml up -d
```

## Security Considerations

### Network Security

```bash
# Use Docker's internal networking
# Services communicate via container names
# Only expose necessary ports to host

# Example: Only dashboard and metrics exposed
ports:
  - "127.0.0.1:8501:8501"  # Bind to localhost only
  - "127.0.0.1:9090:9090"
```

### Container Security

```bash
# All containers run as non-root user (UID 1001)
# Security options enabled:
# - No new privileges
# - Private tmp directories
# - Read-only root filesystem (where applicable)
```

### SSL/TLS Configuration

```bash
# Generate self-signed certificates (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/docker/nginx/ssl/key.pem \
  -out deployment/docker/nginx/ssl/cert.pem

# For production, use certificates from your CA
```

### Secrets Management

```bash
# Use Docker secrets in production
echo "your-db-password" | docker secret create db_password -
echo "your-grafana-password" | docker secret create grafana_password -

# Update docker-compose.yml to use secrets
```

## Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check container logs
docker logs container-analytics-scheduler

# Check resource usage
docker stats

# Check disk space
df -h

# Check if ports are available
netstat -tlnp | grep -E ":(8501|9090|3000|80|443)"
```

#### Image Download Issues

```bash
# Check scheduler logs
docker logs container-analytics-scheduler

# Test Chrome/Selenium manually
docker exec -it container-analytics-scheduler google-chrome --version

# Check network connectivity
docker exec container-analytics-scheduler curl -I https://cdn.draydog.com
```

#### Detection Performance Issues

```bash
# Check GPU availability (if using GPU)
docker exec container-analytics-detector nvidia-smi

# Monitor resource usage
docker stats container-analytics-detector

# Check YOLO model file
docker exec container-analytics-detector ls -la /app/data/models/
```

#### Database Issues

```bash
# Check database file
docker exec container-analytics-scheduler sqlite3 /data/database.db ".tables"

# Check database size and recent activity
docker exec container-analytics-scheduler sqlite3 /data/database.db \
  "SELECT COUNT(*) FROM images WHERE created_at > datetime('now', '-1 hour')"
```

### Performance Tuning

#### CPU Optimization

```bash
# Adjust CPU limits in docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4.0'  # Increase for detection service
    reservations:
      cpus: '1.0'
```

#### Memory Optimization

```bash
# Adjust memory limits
deploy:
  resources:
    limits:
      memory: 8G  # Increase for detection service
```

#### GPU Usage

```bash
# Enable GPU support in docker-compose.yml
deploy:
  resources:
    devices:
      - driver: nvidia
        count: 1
        capabilities: [gpu]
```

### Maintenance Tasks

#### Weekly Maintenance

```bash
# Clean up old images
docker system prune -f

# Restart services for memory cleanup
docker-compose -f deployment/docker/docker-compose.yml restart

# Check log sizes and rotate if needed
find logs/ -name "*.log" -size +100M
```

#### Monthly Maintenance

```bash
# Update base images
docker-compose -f deployment/docker/docker-compose.yml pull
docker-compose -f deployment/docker/docker-compose.yml up -d

# Clean up old data based on retention policy
docker exec container-analytics-scheduler \
  sqlite3 /data/database.db \
  "DELETE FROM images WHERE created_at < datetime('now', '-30 days')"
```

## Support and Monitoring

### Key Metrics to Monitor

1. **Service Health**: All services should show status = 1
2. **Processing Queue**: Should remain < 50 images
3. **Detection Accuracy**: Should remain > 90%
4. **Disk Usage**: Should not exceed 80%
5. **Memory Usage**: Should not exceed configured limits

### Alerting Thresholds

Configure alerts for:
- Service down for > 5 minutes
- Queue size > 100 images for > 30 minutes
- Detection accuracy < 85%
- Disk usage > 90%
- Memory usage > 95% of limits

### Support Channels

- Logs: Check structured logs in `/logs/` directory
- Metrics: Monitor via Prometheus at port 9090
- Dashboards: View Grafana dashboards at port 3000
- Health Checks: Use built-in health check endpoints

For additional support, provide:
1. Service logs from the past hour
2. Current metrics snapshot
3. System resource usage
4. Configuration files (with secrets redacted)

## Scaling and Load Balancing

### Horizontal Scaling

Only the detector service supports horizontal scaling:

```bash
# Scale detector service
docker-compose -f deployment/docker/docker-compose.yml up -d --scale detector=3

# Load balance across multiple detector instances
# Redis is used for coordination between instances
```

### Vertical Scaling

Adjust resource limits based on workload:

```bash
# High-performance configuration
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 16G
    reservations:
      cpus: '2.0'  
      memory: 4G
```

### Geographic Distribution

For multiple locations:

1. Deploy separate stacks at each location
2. Use centralized metrics collection
3. Configure location-specific stream URLs
4. Implement data aggregation service

This completes the comprehensive deployment guide for Container Analytics production infrastructure.
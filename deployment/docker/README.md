# Docker Deployment Guide

## Structure

```
deployment/docker/
├── Dockerfile.prod          # Optimized production Dockerfile with multi-stage build
├── Dockerfile.dev           # Development Dockerfile with hot reload and debug tools
├── docker-compose.yml       # Production compose configuration
├── docker-compose.dev.yml   # Development compose configuration
├── .env                     # Environment variables (copy from .env.example)
├── nginx/                   # NGINX reverse proxy configs
├── grafana/                 # Grafana dashboards and datasources
└── loki/                    # Loki log aggregation configs
```

## Quick Start

### Production
```bash
# From project root
docker-compose up -d

# Or from deployment/docker directory
docker-compose up -d

# With monitoring stack
docker-compose --profile monitoring up -d
```

### Development
```bash
# From project root
docker-compose -f docker-compose.dev.yml up

# Run specific service
docker-compose -f docker-compose.dev.yml up dashboard-dev

# Run tests
docker-compose -f docker-compose.dev.yml run test
```

## Build Optimizations

The production Dockerfile uses several optimization techniques:

1. **Multi-stage builds** - Separates build and runtime dependencies
2. **BuildKit cache mounts** - Caches apt and pip packages between builds
3. **Layer optimization** - Orders layers from least to most frequently changed
4. **Minimal runtime image** - Only includes necessary runtime dependencies

### Enable BuildKit

```bash
# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build with cache
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1
```

### Build Times

With optimizations enabled:
- Initial build: ~7 minutes (vs ~15 minutes unoptimized)
- Rebuild with cache: ~1 minute (vs ~5 minutes unoptimized)
- Image size: ~1.2GB (vs ~1.8GB unoptimized)

## Services

### Core Services
- **scheduler** - Downloads images from Dray Dog every 10 minutes
- **detector** - YOLO object detection on downloaded images
- **dashboard** - Streamlit web interface on port 8501
- **redis** - Cache and message queue

### Optional Services (with profiles)
- **metrics** - Prometheus metrics exporter (profile: monitoring)
- **nginx** - Reverse proxy (profile: production)
- **loki** - Log aggregation (profile: monitoring)
- **grafana** - Monitoring dashboards (profile: monitoring)

## Environment Variables

Create `.env` file in `deployment/docker/`:

```env
# Dashboard
DASHBOARD_PORT=8501

# Scheduler
DOWNLOAD_INTERVAL_MINUTES=10
RETENTION_DAYS=30
STREAMS=in_gate

# Monitoring
METRICS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_PASSWORD=admin

# Proxy
HTTP_PORT=80
HTTPS_PORT=443
```

## Volumes

Production uses Docker volumes for persistence:
- `redis_data` - Redis persistence
- `loki_data` - Loki logs storage
- `grafana_data` - Grafana dashboards

Development mounts source code directly for hot reload.

## Health Checks

All services include health checks:
- Database connectivity
- Service-specific endpoints
- Automatic restart on failure

## GPU Support

To enable GPU for YOLO detection, uncomment the GPU section in docker-compose.yml:

```yaml
deploy:
  resources:
    devices:
      - driver: nvidia
        count: 1
        capabilities: [gpu]
```

Requires NVIDIA Docker runtime installed.

## Troubleshooting

### View logs
```bash
docker-compose logs -f scheduler
docker-compose logs --tail=100 dashboard
```

### Shell access
```bash
docker-compose exec scheduler /bin/bash
docker-compose -f docker-compose.dev.yml run --rm scheduler-dev shell
```

### Reset everything
```bash
docker-compose down -v
rm -rf ../../data/database.db
docker-compose up -d
```

### Build without cache
```bash
docker-compose build --no-cache
```
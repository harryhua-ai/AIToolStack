# Deployment Guide

This guide covers deploying AIToolStack using Docker Compose.

## Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd AIToolStack

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings (see docs/ENV.md)

# 3. Start services
docker compose up -d

# 4. Check health
curl http://localhost:8000/health
```

Access the application at: **http://localhost:8000**

## Services

| Service | Image | Ports | Description |
|---------|-------|-------|-------------|
| camthink | camthink/aitoolstack:latest | 8000:8000 | Main application (frontend + backend) |
| mosquitto | eclipse-mosquitto:2.0 | 1883:1883, 8883:8883 | MQTT broker (MQTT + MQTTS) |

## System Requirements

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Memory**: 4 GB minimum, 8 GB recommended
- **Disk**: 10 GB free space
- **OS**: Linux, macOS, or Windows with WSL2

## Directory Structure

After first run, additional directories will be created:

```
AIToolStack/
├── datasets/          # Dataset storage (auto-created)
├── data/              # Application data (auto-created)
├── backend/data/      # SQLite database (auto-created)
├── mosquitto/
│   ├── config/        # MQTT configuration (auto-created)
│   ├── data/          # MQTT persistence (auto-created)
│   └── log/           # MQTT logs (auto-created)
└── ne301/             # NE301 compiler (auto-cloned)
```

## Docker Volumes

| Host Path | Container Path | Purpose | Persistence |
|-----------|----------------|---------|-------------|
| `./datasets` | `/app/datasets` | Dataset storage | Yes |
| `./data` | `/app/data` | Application data | Yes |
| `backend-data` (volume) | `/app/backend/data` | Database | Yes |
| `./backend` | `/app/backend` | Backend code (dev) | No |
| `./frontend/build` | `/app/frontend/build` | Frontend build (dev) | No |
| `./ne301` | `/workspace/ne301` | NE301 project | Yes |
| `/var/run/docker.sock` | `/var/run/docker.sock` | Docker-in-Docker | N/A |
| `./mosquitto/config` | `/mosquitto/config` | MQTT config | Yes |
| `./mosquitto/data` | `/mosquitto/data` | MQTT data | Yes |
| `./mosquitto/log` | `/mosquitto/log` | MQTT logs | Yes |

## Environment Configuration

### 1. Copy example file
```bash
cp .env.example .env
```

### 2. Edit .env file
Key variables to configure:
```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=False

# MQTT (External Access)
MQTT_BROKER_HOST=192.168.110.106  # Change to your server's IP

# Optional: External Database
# DATABASE_URL=postgresql://user:pass@host:5432/db
```

See [docs/ENV.md](ENV.md) for complete configuration reference.

## Building and Starting

### First Time (Build + Start)
```bash
docker compose up -d
```
This will:
1. Build Docker image (takes 5-10 minutes first time)
2. Start all services
3. Initialize NE301 compiler (auto-cloned)
4. Initialize Mosquitto configuration

### Rebuild After Changes
```bash
# Full rebuild (slowest)
docker compose build --no-cache
docker compose up -d

# Rebuild specific service
docker compose build camthink
docker compose up -d camthink

# Frontend changes only (fastest)
cd frontend && npm run build
docker compose restart camthink
```

### Start/Stop/Restart
```bash
# Start all services
docker compose start

# Stop all services
docker compose stop

# Restart all services
docker compose restart

# Restart specific service
docker compose restart camthink
```

## Health Checks

### Check Service Status
```bash
docker compose ps
```

Expected output:
```
NAME                   STATUS                    PORTS
camthink-aitoolstack   Up X minutes (healthy)    0.0.0.0:8000->8000/tcp
camthink-mosquitto     Up X minutes              0.0.0.0:1883->1883, 8883:8883
```

### API Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "mqtt_enabled": true,
  "mqtt_connected": true
}
```

## Logs

### View All Logs
```bash
docker compose logs -f
```

### View Specific Service
```bash
docker compose logs -f camthink
docker compose logs -f mosquitto
```

### Last 100 Lines
```bash
docker compose logs --tail=100 camthink
```

## Updating

### Update Code
```bash
# Pull latest changes
git pull

# Rebuild if Dockerfile changed
docker compose build

# Restart services
docker compose up -d
```

### Update NE301 Compiler
```bash
# Remove existing NE301
rm -rf ne301/

# Restart service (will auto-clone)
docker compose restart camthink

# Verify
ls ne301/
```

## Backup and Restore

### Backup Data
```bash
# Backup datasets
tar -czf datasets-backup-$(date +%Y%m%d).tar.gz datasets/

# Backup database
cp backend/data/annotator.db backend-backup-$(date +%Y%m%d).db

# Backup Mosquitto config
tar -czf mosquitto-config-backup-$(date +%Y%m%d).tar.gz mosquitto/config/
```

### Restore Data
```bash
# Restore datasets
tar -xzf datasets-backup-20260305.tar.gz

# Restore database
cp backend-backup-20260305.db backend/data/annotator.db

# Restart services
docker compose restart
```

## Troubleshooting

### Container Won't Start

**Check logs**:
```bash
docker compose logs camthink
```

**Common issues**:
- Port 8000 already in use: Change `PORT` in .env
- Database locked: Restart services
- Volume mount error: Check directory permissions

### MQTT Connection Issues

**Symptoms**: "mqtt_connected": false in health check

**Solutions**:
1. Check Mosquitto is running: `docker compose ps mosquitto`
2. Check Mosquitto logs: `docker compose logs mosquitto`
3. Verify MQTT_BROKER_HOST in .env
4. Test connection:
   ```bash
   mosquitto_pub -h localhost -t test -m "hello"
   mosquitto_sub -h localhost -t test
   ```

### Out of Memory

**Increase Docker memory**:
- Docker Desktop: Settings > Resources > Memory > 8 GB
- Linux: No limit by default

### NE301 Compilation Fails

**Check NE301 directory**:
```bash
ls ne301/
```

Should contain NE301 project files. If empty:
```bash
rm -rf ne301/
docker compose restart camthink
```

**Check Docker-in-Docker**:
```bash
docker exec camthink-aitoolstack docker ps
```

Should list containers. If error, check socket mount:
```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

### Performance Issues

**Check resource usage**:
```bash
docker stats
```

**Optimize**:
- Increase Docker memory limit
- Use SSD for datasets
- Reduce MAX_IMAGE_SIZE_MB if needed
- Disable DEBUG mode in production

## Production Deployment

### Security Checklist
- [ ] Change default passwords
- [ ] Use HTTPS (reverse proxy)
- [ ] Set `DEBUG=False`
- [ ] Configure firewall rules
- [ ] Use external MQTT authentication
- [ ] Regular backups

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /mqtt {
        proxy_pass http://localhost:1883;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Auto-start on Boot

```bash
# Edit docker-compose.yml
services:
  camthink:
    restart: unless-stopped
  mosquitto:
    restart: unless-stopped
```

## Uninstall

```bash
# Stop and remove containers
docker compose down

# Remove volumes (WARNING: deletes all data)
docker compose down -v

# Remove images
docker rmi camthink/aitoolstack:latest

# Remove project directory
cd ..
rm -rf AIToolStack
```

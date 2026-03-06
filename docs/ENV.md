# Environment Variables

This document describes all environment variables used in the AIToolStack project.

## Quick Setup

```bash
# Copy the example file
cp .env.example .env

# Edit with your settings
nano .env
```

## Variables

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `HOST` | No | 0.0.0.0 | Server bind address | 0.0.0.0 |
| `PORT` | No | 8000 | Server port number | 8000 |
| `DEBUG` | No | False | Enable debug mode | True/False |
| `MQTT_ENABLED` | No | true | Enable MQTT functionality | true/false |
| `MQTT_USE_BUILTIN_BROKER` | No | true | Use built-in MQTT broker | true/false |
| `MQTT_PORT` | No | 1883 | MQTT broker port | 1883 |
| `MQTT_BUILTIN_PORT` | No | 1883 | Built-in broker port | 1883 |
| `MQTT_BROKER` | Conditional* | - | External broker hostname | mosquitto |
| `MQTT_BROKER_HOST` | Conditional** | - | External broker IP for clients | 192.168.1.100 |
| `MQTT_USERNAME` | No | - | MQTT username (if auth required) | user |
| `MQTT_PASSWORD` | No | - | MQTT password (if auth required) | pass |
| `DATABASE_URL` | No | sqlite:///... | Database connection string | postgres://... |
| `DATASETS_ROOT` | No | /app/datasets | Dataset storage path | /path/to/data |
| `MAX_IMAGE_SIZE_MB` | No | 10 | Max upload size in MB | 10 |

\* Required when `MQTT_USE_BUILTIN_BROKER=false`
\** Should be set to your server's actual IP address for device access

## Configuration Examples

### Development (Default)
```bash
HOST=0.0.0.0
PORT=8000
DEBUG=True
MQTT_ENABLED=true
MQTT_USE_BUILTIN_BROKER=true
```

### Production with External MQTT
```bash
HOST=0.0.0.0
PORT=8000
DEBUG=False
MQTT_ENABLED=true
MQTT_USE_BUILTIN_BROKER=false
MQTT_BROKER=mosquitto
MQTT_BROKER_HOST=192.168.1.100
MQTT_USERNAME=production_user
MQTT_PASSWORD=secure_password
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

### Docker Compose
Most variables are set in `docker-compose.yml`. Override in `.env`:
```bash
# .env file
MQTT_BROKER_HOST=192.168.110.106
DATABASE_URL=postgresql://user:pass@db:5432/aitoolstack
```

## MQTT Configuration Details

### Built-in Broker (Default)
When `MQTT_USE_BUILTIN_BROKER=true`, the application runs an embedded MQTT broker:
- Simpler setup, no external dependencies
- Suitable for development and small deployments
- Runs on port specified by `MQTT_BUILTIN_PORT` (default: 1883)

### External Broker
When using external Mosquitto broker:
```bash
MQTT_USE_BUILTIN_BROKER=false
MQTT_BROKER=mosquitto  # Docker service name or hostname
MQTT_BROKER_HOST=192.168.110.106  # External IP for client connections
```

**Important**: `MQTT_BROKER` is used internally (server-to-broker), while `MQTT_BROKER_HOST` is exposed to clients (browser-to-broker).

## Database Configuration

### SQLite (Default)
```bash
DATABASE_URL=sqlite:////app/backend/data/annotator.db
```
- No additional setup required
- Suitable for development and small deployments
- File persisted in Docker volume

### PostgreSQL
```bash
DATABASE_URL=postgresql://user:password@host:5432/database
```
- Better for production
- Requires PostgreSQL server
- Update connection details as needed

## File Upload Limits

```bash
MAX_IMAGE_SIZE_MB=10
```
- Maximum size for image uploads
- Default: 10 MB
- Adjust based on your needs

## Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
lsof -i :8000  # For PORT
lsof -i :1883  # For MQTT_PORT
```

### MQTT Connection Failed
1. Check broker is running: `docker compose ps`
2. Verify `MQTT_BROKER_HOST` is accessible from clients
3. Test connection: `mosquitto_pub -h <host> -t test -m hello`
4. Check logs: `docker compose logs mosquitto`

### Database Connection Error
- Verify `DATABASE_URL` format
- Check database server is running
- Ensure network access to database host

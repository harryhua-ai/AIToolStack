"""MQTT service: subscribe to images uploaded by devices"""
import json
import base64
import uuid
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import deque
import paho.mqtt.client as mqtt
from PIL import Image as PILImage
import io
import hashlib

from backend.config import settings
from backend.models.database import SessionLocal, Image, Project
from backend.services.websocket_manager import websocket_manager
from backend.services.mqtt_broker import builtin_mqtt_broker
from backend.services.mqtt_config_service import MQTTConfig, mqtt_config_service
from backend.services.external_broker_service import external_broker_service

logger = logging.getLogger(__name__)


class MQTTService:
    """MQTT subscription service"""
    
    def __init__(self, config_service=mqtt_config_service):
        # For backward compatibility, self.client keeps a reference to the
        # first created MQTT client (primary broker). When multi-broker
        # support is enabled (mode == \"both\"), additional clients are stored
        # in self.clients and callbacks receive the actual client instance.
        self.client: Optional[mqtt.Client] = None
        self.clients: list[mqtt.Client] = []
        # Track last known broker endpoints and their connection states.
        # Each item: {"type": "builtin"|"external", "host": str, "port": int, "connected": bool}
        self._endpoints: list[dict] = []
        self.is_connected = False
        # For multi-broker scenarios, these fields reflect the *primary* broker
        # (the first one in the list); logging for each client uses per-client
        # attributes instead.
        self.broker_host = ""  # Save current connected broker address (primary)
        self.broker_port = 0
        self._config_service = config_service
        self._config: Optional[MQTTConfig] = None
        
        # Connection statistics and monitoring
        self.connection_count = 0
        self.disconnection_count = 0
        self.last_connect_time: Optional[float] = None
        self.last_disconnect_time: Optional[float] = None
        self.recent_errors = deque(maxlen=10)  # Keep last 10 errors
        self.message_count = 0
        self.last_message_time: Optional[float] = None
        
        # Deduplication: Track processed messages to prevent duplicate image uploads
        # Format: {message_id: timestamp}
        # Messages older than 1 hour are automatically removed
        self._processed_messages: dict[str, float] = {}
        self._dedup_cleanup_interval = 3600  # 1 hour in seconds
    
    def on_connect(self, client, userdata, flags, rc):
        """Connection callback"""
        if rc == 0:
            self.is_connected = True
            self.connection_count += 1
            self.last_connect_time = time.time()
            # Prefer per-client broker metadata if available
            broker_host = getattr(client, "_camthink_broker_host", self.broker_host)
            broker_port = getattr(client, "_camthink_broker_port", self.broker_port)
            broker_index = getattr(client, "_camthink_broker_index", None)
            if isinstance(broker_index, int) and 0 <= broker_index < len(self._endpoints):
                self._endpoints[broker_index]["connected"] = True
            logger.info(f"Connected to broker at {broker_host}:{broker_port}")
            # Subscribe to topic pattern (broker-specific or default)
            topic_pattern = getattr(client, "_camthink_topic_pattern", settings.MQTT_UPLOAD_TOPIC)
            broker_qos = getattr(client, "_camthink_broker_qos", settings.MQTT_QOS)
            try:
                result = client.subscribe(topic_pattern, qos=broker_qos)
                if result[0] == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"Subscribed to topic pattern: {topic_pattern} (QoS: {broker_qos})")
                else:
                    logger.error(f"Failed to subscribe to topic {topic_pattern}: error code {result[0]}")
            except Exception as e:
                logger.error(f"Error subscribing to topic: {e}")
        else:
            error_msg = self._get_connection_error_message(rc)
            logger.error(f"Connection failed with code {rc}: {error_msg}")
            self.recent_errors.append({
                'time': time.time(),
                'type': 'connect_error',
                'code': rc,
                'message': error_msg
            })
    
    def on_disconnect(self, client, userdata, rc):
        """Disconnect callback"""
        broker_index = getattr(client, "_camthink_broker_index", None)
        if isinstance(broker_index, int) and 0 <= broker_index < len(self._endpoints):
            self._endpoints[broker_index]["connected"] = False
        # Recompute overall connection flag based on all endpoints
        # If any broker is connected, consider service as connected
        self.is_connected = any(ep.get("connected", False) for ep in self._endpoints)
        self.disconnection_count += 1
        self.last_disconnect_time = time.time()
        
        if rc != 0:
            # Abnormal disconnect - paho-mqtt will automatically try to reconnect
            error_msg = self._get_disconnect_error_message(rc)
            broker_host = getattr(client, "_camthink_broker_host", self.broker_host)
            broker_port = getattr(client, "_camthink_broker_port", self.broker_port)
            logger.warning(f"Disconnected from broker {broker_host}:{broker_port} unexpectedly (rc={rc}): {error_msg}")
            self.recent_errors.append({
                'time': time.time(),
                'type': 'disconnect_error',
                'code': rc,
                'message': error_msg
            })
        else:
            # Normal disconnect
            logger.info("Disconnected from broker normally")
        
        # If abnormal disconnect (rc != 0), paho-mqtt will automatically try to reconnect
        # We don't need to manually handle reconnection logic
    
    def on_message(self, client, userdata, msg):
        """Message receive callback"""
        try:
            self.message_count += 1
            self.last_message_time = time.time()
            
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            # Parse topic to get project_id
            # Topic format: annotator/upload/{project_id}
            parts = topic.split('/')
            if len(parts) < 3:
                logger.warning(f"Invalid topic format: {topic}")
                return
            
            project_id = parts[2]
            
            # Parse JSON payload
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for topic {topic}: {e}")
                # Try to extract req_id and device_id from raw payload
                req_id = ''
                device_id = ''
                try:
                    temp_data = json.loads(payload)  # This will fail, but we try
                except Exception:
                    pass
                self._send_error_response(client, req_id, device_id, "Invalid JSON format")
                return
            
            # Handle image upload
            self._handle_image_upload(client, project_id, data, topic)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            req_id = ''
            device_id = ''
            try:
                data = json.loads(msg.payload.decode('utf-8'))
                req_id = data.get('req_id', '')
                device_id = data.get('device_id', '')
            except:
                pass
            self._send_error_response(client, req_id, device_id, str(e))
    
    def _get_message_id(self, data: dict, topic: str) -> str:
        """Generate a unique message ID for deduplication.
        
        Uses req_id if available, otherwise creates a hash from topic + payload.
        """
        req_id = data.get('req_id')
        if req_id:
            return f"req_{req_id}"
        
        # Fallback: create hash from topic + device_id + timestamp
        device_id = data.get('device_id', 'unknown')
        timestamp = data.get('timestamp', '')
        # For new format, use image_id from metadata if available
        if 'image_data' in data:
            metadata = data.get('metadata', {})
            image_id = metadata.get('image_id', '')
            if image_id:
                return f"img_{image_id}"
        
        # Last resort: hash of topic + device_id + timestamp
        content = f"{topic}:{device_id}:{timestamp}"
        return f"hash_{hashlib.md5(content.encode()).hexdigest()}"
    
    def _is_duplicate_message(self, message_id: str) -> bool:
        """Check if message has already been processed.
        
        Also performs cleanup of old entries to prevent memory growth.
        """
        current_time = time.time()
        
        # Cleanup old entries (older than 1 hour)
        if len(self._processed_messages) > 1000:  # Only cleanup when dict is large
            cutoff_time = current_time - self._dedup_cleanup_interval
            self._processed_messages = {
                msg_id: ts for msg_id, ts in self._processed_messages.items()
                if ts > cutoff_time
            }
        
        # Check if message was already processed
        if message_id in self._processed_messages:
            return True
        
        # Mark as processed
        self._processed_messages[message_id] = current_time
        return False
    
    def _handle_image_upload(self, client, project_id: str, data: dict, topic: str):
        """Handle image upload"""
        # Check for duplicate messages to prevent processing the same image multiple times
        # This can happen when multiple MQTT clients subscribe to the same topic
        message_id = self._get_message_id(data, topic)
        if self._is_duplicate_message(message_id):
            logger.warning(f"Duplicate message detected (message_id: {message_id}), skipping processing")
            # Still send success response to avoid device retries
            req_id = data.get('req_id', '')
            device_id = data.get('device_id', 'unknown')
            self._send_success_response(client, req_id, device_id, project_id)
            return
        
        # Adapt to new data structure
        # Support two formats:
        # 1. New format: { "image_data": "...", "encoding": "...", "metadata": {...} }
        # 2. Old format: { "req_id": "...", "device_id": "...", "image": {...} }
        
        # Try new format
        if 'image_data' in data:
            # New format
            req_id = data.get('req_id', str(uuid.uuid4()))
            device_id = data.get('device_id', topic.split('/')[-1] if '/' in topic else 'unknown')
            metadata = data.get('metadata', {})
            encoding = data.get('encoding', 'base64')
            base64_data = data.get('image_data', '')
            
            # Extract information from metadata
            image_id = metadata.get('image_id', f'img_{int(datetime.utcnow().timestamp())}')
            timestamp = metadata.get('timestamp', int(datetime.utcnow().timestamp()))
            image_format = metadata.get('format', 'jpeg').lower()
            # If metadata has dimension information, use it first
            metadata_width = metadata.get('width')
            metadata_height = metadata.get('height')
            
            # Generate filename
            if image_format in ['jpeg', 'jpg']:
                filename = f'{image_id}.jpg'
            elif image_format == 'png':
                filename = f'{image_id}.png'
            else:
                filename = f'{image_id}.{image_format}'
        else:
            # Old format (backward compatible)
            req_id = data.get('req_id', str(uuid.uuid4()))
            device_id = data.get('device_id', 'unknown')
            timestamp = data.get('timestamp', int(datetime.utcnow().timestamp()))
            image_data = data.get('image', {})
            metadata = data.get('metadata', {})
            
            filename = image_data.get('filename', f'img_{timestamp}.jpg')
            image_format = image_data.get('format', 'jpg').lower()
            encoding = image_data.get('encoding', 'base64')
            base64_data = image_data.get('data', '')
            metadata_width = None
            metadata_height = None
        
        # Verify project exists
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                error_msg = f"Project {project_id} not found"
                logger.warning(error_msg)
                self._send_error_response(client, req_id, device_id, error_msg)
                return
            
            # Process base64 data, remove possible data URI prefix
            # Supported formats:
            # 1. data:image/jpeg;base64,xxxxx
            # 2. data:image/png;base64,xxxxx
            # 3. data:image/jpg;base64,xxxxx
            # 4. Pure base64 string
            if base64_data.startswith('data:'):
                # Contains data URI prefix, extract base64 part
                if ',' in base64_data:
                    base64_data = base64_data.split(',')[-1]
                else:
                    # If format is abnormal, try to remove data: prefix
                    base64_data = base64_data.replace('data:', '').split(';')[-1]
            elif ',' in base64_data:
                # Might contain other separators
                base64_data = base64_data.split(',')[-1]
            
            # Clean possible whitespace characters
            base64_data = base64_data.strip()
            
            # Base64 decode
            if encoding != 'base64':
                raise ValueError(f"Unsupported encoding: {encoding}")
            
            try:
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 data: {str(e)}")
            
            # Verify image size
            size_mb = len(image_bytes) / (1024 * 1024)
            if size_mb > settings.MAX_IMAGE_SIZE_MB:
                raise ValueError(f"Image too large: {size_mb:.2f}MB (max: {settings.MAX_IMAGE_SIZE_MB}MB)")
            
            # Get image dimensions
            if metadata_width and metadata_height:
                # Use dimension information from metadata
                img_width = metadata_width
                img_height = metadata_height
            else:
                # Open image to get dimensions
                img = PILImage.open(io.BytesIO(image_bytes))
                img_width, img_height = img.size
            
            # Generate storage path
            project_dir = settings.DATASETS_ROOT / project_id / "raw"
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle filename conflicts
            file_path = project_dir / filename
            if file_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                timestamp_suffix = int(datetime.utcnow().timestamp())
                filename = f"{stem}_{timestamp_suffix}{suffix}"
                file_path = project_dir / filename
            
            # Save image
            file_path.write_bytes(image_bytes)
            
            # Generate relative path (only includes raw/filename, not project_id)
            relative_path = f"raw/{filename}"
            
            # Save to database
            db_image = Image(
                project_id=project_id,
                filename=filename,
                path=relative_path,
                width=img_width,
                height=img_height,
                status="UNLABELED",
                source=f"MQTT:{device_id}"
            )
            db.add(db_image)
            db.commit()
            db.refresh(db_image)
            
            # Ensure database transaction is fully committed before notifying frontend
            # This prevents frontend from refreshing before the new image is visible in the database
            image_id = db_image.id
            
            logger.info(f"Image saved: {filename} ({img_width}x{img_height}) to project {project_id}, image_id: {image_id}")
            
            # Send success response
            self._send_success_response(client, req_id, device_id, project_id)
            
            # Notify frontend via WebSocket (after database commit is complete)
            try:
                websocket_manager.broadcast_project_update(project_id, {
                    "type": "new_image",
                    "image_id": image_id,
                    "filename": filename,
                    "path": relative_path,
                    "width": img_width,
                    "height": img_height
                })
                logger.debug(f"WebSocket notification sent for new image {image_id} in project {project_id}")
            except Exception as ws_error:
                logger.error(f"Failed to send WebSocket notification for new image: {ws_error}", exc_info=True)
                # Don't fail the whole operation if WebSocket notification fails
            
        except Exception as e:
            db.rollback()
            error_msg = f"Failed to save image: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._send_error_response(client, req_id, device_id, error_msg)
        finally:
            db.close()
    
    def _send_success_response(self, client, req_id: str, device_id: str, project_id: str):
        """Send success response"""
        if not device_id or device_id == 'unknown':
            return
        
        if not client or not self.is_connected:
            logger.warning(f"Cannot send success response: client not connected")
            return
        
        # Get broker-specific QoS
        broker_type = getattr(client, "_camthink_broker_type", "builtin")
        if self._config:
            qos = self._config.builtin_qos if broker_type == "builtin" else self._config.external_qos
        else:
            qos = settings.MQTT_QOS
        
        response_topic = f"{settings.MQTT_RESPONSE_TOPIC_PREFIX}/{device_id}"
        response = {
            "req_id": req_id,
            "status": "success",
            "code": 200,
            "message": f"Image saved to project {project_id}",
            "server_time": int(datetime.utcnow().timestamp())
        }
        
        try:
            result = client.publish(response_topic, json.dumps(response), qos=qos)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(f"Failed to publish success response: error code {result.rc}")
        except Exception as e:
            logger.error(f"Error publishing success response: {e}")
    
    def _send_error_response(self, client, req_id: str, device_id: str, error_message: str):
        """Send error response"""
        if not device_id or device_id == 'unknown':
            return
        
        if not client or not self.is_connected:
            logger.warning(f"Cannot send error response: client not connected")
            return
        
        # Get broker-specific QoS
        broker_type = getattr(client, "_camthink_broker_type", "builtin")
        if self._config:
            qos = self._config.builtin_qos if broker_type == "builtin" else self._config.external_qos
        else:
            qos = settings.MQTT_QOS
        
        response_topic = f"{settings.MQTT_RESPONSE_TOPIC_PREFIX}/{device_id}"
        response = {
            "req_id": req_id,
            "status": "error",
            "code": 400,
            "message": error_message,
            "server_time": int(datetime.utcnow().timestamp())
        }
        
        try:
            result = client.publish(response_topic, json.dumps(response), qos=qos)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(f"Failed to publish error response: error code {result.rc}")
        except Exception as e:
            logger.error(f"Error publishing error response: {e}")
    
    def _get_connection_error_message(self, rc: int) -> str:
        """Get human-readable connection error message"""
        error_messages = {
            1: "Connection refused - incorrect protocol version",
            2: "Connection refused - invalid client identifier",
            3: "Connection refused - server unavailable",
            4: "Connection refused - bad username or password",
            5: "Connection refused - not authorized"
        }
        return error_messages.get(rc, f"Unknown error code {rc}")
    
    def _get_disconnect_error_message(self, rc: int) -> str:
        """Get human-readable disconnect error message"""
        # rc values for on_disconnect:
        # 0 = normal disconnect
        # Non-zero = unexpected disconnect
        # Common values: network error, timeout, etc.
        if rc == 0:
            return "Normal disconnect"
        elif rc == 7:
            return "Network error or timeout - connection may have timed out"
        else:
            return f"Unexpected disconnect (error code: {rc})"
    
    def get_status(self) -> dict:
        """Get current MQTT service status"""
        return {
            'connected': self.is_connected,
            'broker': f"{self.broker_host}:{self.broker_port}" if self.broker_host else None,
            'brokers': [
                {
                    'type': ep.get("type"),
                    'host': ep.get("host"),
                    'port': ep.get("port"),
                    'broker_id': ep.get("broker_id"),  # For external brokers
                    'connected': ep.get("connected", False),
                }
                for ep in self._endpoints
            ],
            'connection_count': self.connection_count,
            'disconnection_count': self.disconnection_count,
            'message_count': self.message_count,
            'last_connect_time': self.last_connect_time,
            'last_disconnect_time': self.last_disconnect_time,
            'last_message_time': self.last_message_time,
            'recent_errors': list(self.recent_errors)
        }
    
    def start(self):
        """Start MQTT client using runtime configuration."""
        print("[MQTT Service] ===== start() method called =====")
        logger.info("[MQTT Service] ===== start() method called =====")
        # Load current config
        self._config = self._config_service.load_config()
        print(f"[MQTT Service] Config loaded: enabled={self._config.enabled}, builtin_protocol={self._config.builtin_protocol}")
        logger.info(f"[MQTT Service] Config loaded: enabled={self._config.enabled}, builtin_protocol={self._config.builtin_protocol}")
        # Reset previous clients list
        self.clients = []
        self._endpoints = []

        if not self._config.enabled:
            logger.info("MQTT service is disabled in configuration")
            return
        
        try:
            cfg = self._config

            # Build a list of broker endpoints to connect to
            endpoints = []
            # Built-in broker endpoint used by the training/annotation service.
            # 当使用 Python 内置 aMQTT 时，指向容器内的 127.0.0.1:MQTT_BUILTIN_PORT；
            # 当关闭内置 broker（MQTT_USE_BUILTIN_BROKER=false）时，把“内置”端点指向外部 broker，
            # 这样前端的“内置 MQTT Broker 状态”就反映我们实际使用的 Mosquitto/外部服务。
            from backend.config import settings as app_settings  # 避免循环引用

            # Client connection automatically infers protocol and port from broker configuration
            # - If broker_protocol is "mqtts", client connects to TLS port (8883)
            # - If broker_protocol is "mqtt", client connects to TCP port (1883)
            # This ensures client always matches broker's actual configuration
            if cfg.builtin_protocol == "mqtts":
                # Broker is configured for MQTTS, client connects to TLS port
                builtin_port = cfg.builtin_tls_port or 8883
                client_protocol = "mqtts"
            else:
                # Broker is configured for MQTT, client connects to TCP port
                builtin_port = cfg.builtin_tcp_port or app_settings.MQTT_BUILTIN_PORT
                client_protocol = "mqtt"

            if app_settings.MQTT_USE_BUILTIN_BROKER:
                builtin_host = "127.0.0.1"
            else:
                # 不使用内置 broker 时，将“内置”连接指向外部 broker（例如 mosquitto）
                # 在 docker-compose 中通过 MQTT_BROKER=mosquitto 注入
                builtin_host = app_settings.MQTT_BROKER or "localhost"

            endpoints.append(
                {
                    "type": "builtin",
                    "host": builtin_host,
                    "port": builtin_port,
                    "protocol": client_protocol,  # Client protocol matches broker protocol automatically
                    "connected": False,
                }
            )
            # Multiple external brokers from database
            try:
                external_brokers = external_broker_service.get_enabled_brokers()
                for broker in external_brokers:
                    endpoints.append(
                        {
                            "type": "external",
                            "host": broker.host,
                            "port": broker.port,
                            "protocol": broker.protocol,
                            "connected": False,
                            "broker_id": broker.id,
                            "broker_name": broker.name,
                            "username": broker.username,
                            "password": broker.password,
                            "qos": broker.qos,
                            "keepalive": broker.keepalive,
                            "tls_enabled": broker.tls_enabled,
                            "tls_ca_cert_path": broker.tls_ca_cert_path,
                            "tls_client_cert_path": broker.tls_client_cert_path,
                            "tls_client_key_path": broker.tls_client_key_path,
                            # tls_insecure_skip_verify is not applicable - AIToolStack as client should always verify
                            # topic_pattern is None - will use default from settings in MQTT service
                            "topic_pattern": None,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load external brokers from database: {e}")
                # Fallback to legacy single external broker config
                if cfg.external_enabled and (cfg.external_host or cfg.external_port):
                    endpoints.append(
                        {
                            "type": "external",
                            "host": cfg.external_host or settings.MQTT_BROKER,
                            "port": cfg.external_port or settings.MQTT_PORT,
                            "protocol": cfg.external_protocol,
                            "connected": False,
                            "broker_id": None,
                            "broker_name": "Legacy External Broker",
                            "username": cfg.external_username,
                            "password": cfg.external_password,
                            "qos": cfg.external_qos,
                            "keepalive": cfg.external_keepalive,
                            "tls_enabled": cfg.external_tls_enabled,
                            "tls_ca_cert_path": cfg.external_tls_ca_cert_path,
                            "tls_client_cert_path": cfg.external_tls_client_cert_path,
                            "tls_client_key_path": cfg.external_tls_client_key_path,
                            # tls_insecure_skip_verify is not applicable - AIToolStack as client should always verify
                            # topic_pattern is None - will use default from settings in MQTT service
                            "topic_pattern": None,
                        }
                    )

            if not endpoints:
                logger.warning("MQTT config resulted in no endpoints to connect to")
                return

            # Initialize endpoint tracking
            self._endpoints = endpoints.copy()

            # Keep the first endpoint as primary for backward compatible fields
            primary = endpoints[0]
            self.broker_host = primary["host"]
            self.broker_port = primary["port"]

            # Create and connect a client for each endpoint
            import ssl

            for idx, ep in enumerate(endpoints):
                print(f"[MQTT Service] Creating client for endpoint {idx}: type={ep['type']}, host={ep['host']}, port={ep['port']}, protocol={ep.get('protocol', 'N/A')}")
                logger.info(f"[MQTT Service] Creating client for endpoint {idx}: type={ep['type']}, host={ep['host']}, port={ep['port']}, protocol={ep.get('protocol', 'N/A')}")
                client = mqtt.Client(
                    client_id=f"annotator_server_{ep['type']}_{uuid.uuid4().hex[:8]}",
                    protocol=mqtt.MQTTv311,
                    clean_session=True,
                )

                # Attach endpoint info for logging
                setattr(client, "_camthink_broker_host", ep["host"])
                setattr(client, "_camthink_broker_port", ep["port"])
                setattr(client, "_camthink_broker_type", ep["type"])
                setattr(client, "_camthink_broker_index", idx)
                # Store broker_id for external brokers
                if ep.get("broker_id") is not None:
                    setattr(client, "_camthink_broker_id", ep["broker_id"])
                # Store broker-specific QoS
                if ep["type"] == "builtin":
                    setattr(client, "_camthink_broker_qos", cfg.builtin_qos)
                else:
                    setattr(client, "_camthink_broker_qos", ep.get("qos", cfg.external_qos))

                # Set connection timeout and retry parameters
                client.reconnect_delay_set(min_delay=1, max_delay=120)

                # Set callbacks
                client.on_connect = self.on_connect
                client.on_disconnect = self.on_disconnect
                client.on_message = self.on_message
            
                # Configure broker-specific settings
                if ep["type"] == "builtin":
                    # Built-in broker configuration
                    # 当 MQTT_USE_BUILTIN_BROKER=true 时，通常是 Python 内置 aMQTT；
                    # 当 MQTT_USE_BUILTIN_BROKER=false 时，“builtin” 端点会被指向外部 Mosquitto，
                    # 这里统一负责设置认证信息。
                    protocol = ep.get("protocol", cfg.builtin_protocol)  # Use protocol from endpoint (already inferred from broker config)
                    keepalive = cfg.builtin_keepalive or 120  # Client-side keepalive (independent of broker)
                    topic_pattern = settings.MQTT_UPLOAD_TOPIC
                    print(f"[Client Connection] Connecting to built-in broker: protocol={protocol}, port={ep['port']}, host={ep['host']}")
                    logger.info(f"[Client Connection] Connecting to built-in broker: protocol={protocol}, port={ep['port']}, host={ep['host']}")

                    # Authentication: Client automatically uses broker's authentication settings
                    # - If broker allows anonymous: client connects anonymously
                    # - If broker requires auth: client uses broker's username/password (same as external devices)
                    if not cfg.builtin_allow_anonymous:
                        from backend.config import settings as app_settings
                        builtin_username = cfg.builtin_username or getattr(app_settings, "MQTT_USERNAME", None)
                        builtin_password = cfg.builtin_password or getattr(app_settings, "MQTT_PASSWORD", None)
                        if builtin_username and builtin_password:
                            client.username_pw_set(builtin_username, builtin_password)
                            print(f"[Client Connection] Using broker authentication: username={builtin_username}")
                            logger.info(f"[Client Connection] Using broker authentication: username={builtin_username}")
                        else:
                            print(f"[Client Connection] WARNING: Broker requires authentication but no credentials configured")
                            logger.warning("[Client Connection] Broker requires authentication but no credentials configured")
                    else:
                        print(f"[Client Connection] Broker allows anonymous, connecting without credentials")
                        logger.info("[Client Connection] Broker allows anonymous, connecting without credentials")

                    # TLS configuration: Client automatically uses broker's TLS configuration
                    # - If broker_protocol is "mqtts", client connects via TLS
                    # - Client uses broker's CA certificate to verify server certificate
                    # - Client uses broker's client certificate/key if configured (for mTLS)
                    if protocol == "mqtts":
                        print(f"[Client TLS] ===== Configuring TLS for client connection (broker protocol: {protocol}, port: {ep['port']}) =====")
                        logger.info(f"[Client TLS] Configuring TLS for client connection (broker protocol: {protocol}, port: {ep['port']})")
                        # Client uses broker's CA certificate to verify server certificate
                        # Since AIToolStack and broker are deployed together, we can access the CA file directly
                        tls_kwargs = {
                            "tls_version": ssl.PROTOCOL_TLSv1_2
                        }
                        
                        # Use CA certificate for certificate verification (required for security)
                        if cfg.builtin_tls_ca_cert_path:
                            # Check if CA certificate file exists
                            import os
                            ca_path = cfg.builtin_tls_ca_cert_path
                            print(f"[TLS Config] CA certificate path from config: {ca_path}")
                            if os.path.exists(ca_path):
                                tls_kwargs["ca_certs"] = ca_path
                                tls_kwargs["cert_reqs"] = ssl.CERT_REQUIRED  # Require certificate verification
                                print(f"[TLS Config] ✓ CA certificate found, setting cert_reqs=CERT_REQUIRED")
                                print(f"[TLS Config] ✓ ca_certs={ca_path}")
                                logger.info(f"[TLS Config] Using CA certificate for verification: {ca_path}")
                            else:
                                error_msg = f"CA certificate file not found: {ca_path}"
                                print(f"[TLS Config] ✗ ERROR: {error_msg}")
                                logger.error(f"[TLS Config] {error_msg}")
                                raise FileNotFoundError(error_msg)
                        else:
                            error_msg = "No CA certificate path configured, cannot verify server certificate"
                            print(f"[TLS Config] ✗ ERROR: {error_msg}")
                            logger.error(f"[TLS Config] {error_msg}")
                            raise ValueError("CA certificate path is required for MQTTS connection")
                        
                        # Use default client certificates if available (AIToolStack always uses default client certs)
                        default_client_cert = Path("/mosquitto/config/certs/client.crt")
                        default_client_key = Path("/mosquitto/config/certs/client.key")
                        if default_client_cert.exists() and default_client_key.exists():
                            tls_kwargs["certfile"] = str(default_client_cert)
                            tls_kwargs["keyfile"] = str(default_client_key)
                            print(f"[TLS Config] Using default client certificates: {default_client_cert}")
                            logger.info(f"[TLS Config] Using default client certificates: {default_client_cert}")
                        else:
                            print(f"[TLS Config] No client certificates found, using CA verification only (secure mode)")
                            logger.info("[TLS Config] Using CA certificate verification (secure mode)")
                        
                        # AIToolStack as client should always verify server certificate
                        use_insecure = False
                        
                        # Configure TLS with CA certificate verification
                        # CRITICAL: For security, we MUST verify the server certificate using the CA
                        # Do NOT allow insecure mode to bypass certificate verification
                        print(f"[TLS Config] Calling tls_set with keys: {list(tls_kwargs.keys())}")
                        print(f"[TLS Config] cert_reqs={tls_kwargs.get('cert_reqs')}, ca_certs={tls_kwargs.get('ca_certs')}")
                        logger.info(f"[TLS Config] Calling tls_set with: {list(tls_kwargs.keys())}")
                        logger.info(f"[TLS Config] cert_reqs={tls_kwargs.get('cert_reqs')}, ca_certs={tls_kwargs.get('ca_certs')}")
                        try:
                            client.tls_set(**tls_kwargs)
                            print(f"[TLS Config] ✓ tls_set() completed successfully")
                            logger.info(f"[TLS Config] tls_set() completed successfully with CA certificate verification")
                        except Exception as tls_err:
                            print(f"[TLS Config] ✗ tls_set() failed: {tls_err}")
                            logger.error(f"[TLS Config] tls_set() failed: {tls_err}", exc_info=True)
                            raise
                        
                        # CRITICAL: Always disable insecure mode to enforce certificate verification
                        # Even if user sets insecure_skip_verify, we enforce verification for built-in broker
                        # This ensures only certificates signed by the correct CA can connect
                        use_insecure = False  # Force secure mode for built-in broker
                        print(f"[TLS Config] FORCING tls_insecure_set(False) to enable certificate verification")
                        logger.info(f"[TLS Config] Setting TLS insecure mode: {use_insecure} (FORCED to False for security)")
                        try:
                            client.tls_insecure_set(use_insecure)
                            print(f"[TLS Config] ✓ tls_insecure_set(False) completed successfully")
                            print(f"[TLS Config] ✓ Certificate verification is ENABLED - only certificates signed by CA will be accepted")
                            logger.info(f"[TLS Config] tls_insecure_set({use_insecure}) completed successfully")
                            logger.info(f"[TLS Config] Certificate verification is ENABLED - only certificates signed by CA will be accepted")
                        except Exception as tls_err:
                            print(f"[TLS Config] ✗ tls_insecure_set() failed: {tls_err}")
                            logger.error(f"[TLS Config] tls_insecure_set() failed: {tls_err}", exc_info=True)
                            raise
                        print(f"[TLS Config] ===== TLS configuration completed =====")
                        # Double-check: Verify TLS settings are correct before connecting
                        # Note: paho-mqtt doesn't provide getters, so we can't verify the actual state
                        # But we log what we expect to ensure the code path is correct
                        print(f"[TLS Config] FINAL STATE: cert_reqs=CERT_REQUIRED({ssl.CERT_REQUIRED}), ca_certs={tls_kwargs.get('ca_certs')}, insecure=False")
                    else:
                        logger.info(f"[TLS Config] Protocol is '{protocol}', skipping TLS configuration")
                else:
                    # External broker configuration (from database or legacy config)
                    protocol = ep.get("protocol", cfg.external_protocol)
                    keepalive = ep.get("keepalive", cfg.external_keepalive or 120)
                    # All external brokers use default topic pattern based on system business logic
                    topic_pattern = None  # Will use default from settings.MQTT_UPLOAD_TOPIC
                    # External broker authentication
                    username = ep.get("username") or cfg.external_username
                    password = ep.get("password") or cfg.external_password
                    if username and password:
                        client.username_pw_set(username, password)

                    # TLS configuration for external broker
                    tls_enabled = ep.get("tls_enabled", False) or (protocol == "mqtts" and cfg.external_tls_enabled)
                    if protocol == "mqtts" and tls_enabled:
                        tls_kwargs = {}
                        tls_ca_cert = ep.get("tls_ca_cert_path") or cfg.external_tls_ca_cert_path
                        tls_client_cert = ep.get("tls_client_cert_path") or cfg.external_tls_client_cert_path
                        tls_client_key = ep.get("tls_client_key_path") or cfg.external_tls_client_key_path
                        
                        if tls_ca_cert:
                            tls_kwargs["ca_certs"] = tls_ca_cert
                        if tls_client_cert and tls_client_key:
                            tls_kwargs["certfile"] = tls_client_cert
                            tls_kwargs["keyfile"] = tls_client_key

                        # Use TLS v1.2 as a safe default
                        tls_kwargs["tls_version"] = ssl.PROTOCOL_TLSv1_2

                        if tls_kwargs:
                            client.tls_set(**tls_kwargs)
                        # AIToolStack as client should always verify server certificate
                        # tls_insecure_skip_verify is not applicable for external brokers
                        client.tls_insecure_set(False)

                # Store topic pattern for this client (use default if None)
                # All brokers subscribe to the same default topic pattern based on system business logic
                setattr(client, "_camthink_topic_pattern", topic_pattern or settings.MQTT_UPLOAD_TOPIC)

                # For builtin broker with MQTTS, verify TLS settings one more time before connecting
                if ep["type"] == "builtin" and protocol == "mqtts":
                    print(f"[MQTT Service] Pre-connect verification: Protocol=mqtts, expecting certificate verification")
                    # Ensure insecure mode is still False (in case it was changed elsewhere)
                    client.tls_insecure_set(False)
                    print(f"[MQTT Service] Re-confirmed: tls_insecure_set(False) before connect")
                
                print(f"[MQTT Service] Connecting to {ep['type']} MQTT Broker (protocol: {protocol}) at {ep['host']}:{ep['port']}")
                logger.info(f"[MQTT Service] Connecting to {ep['type']} MQTT Broker (protocol: {protocol}) at {ep['host']}:{ep['port']}")
                try:
                    client.connect(ep["host"], ep["port"], keepalive=keepalive)
                    print(f"[MQTT Service] ✓ connect() call completed for {ep['type']} broker")
                    logger.info(f"[MQTT Service] connect() call completed, starting loop for {ep['type']} broker")
                    client.loop_start()
                    print(f"[MQTT Service] ✓ loop_start() completed for {ep['type']} broker")
                    logger.info(f"[MQTT Service] loop_start() completed for {ep['type']} broker")
                except Exception as conn_err:
                    print(f"[MQTT Service] ✗ Error during connect/loop_start for {ep['type']} broker: {conn_err}")
                    logger.error(f"[MQTT Service] Error during connect/loop_start for {ep['type']} broker: {conn_err}", exc_info=True)
                    raise

                # Save clients for later stop / status
                if self.client is None:
                    self.client = client
                self.clients.append(client)

            logger.info("MQTT client(s) loop started")
        except ConnectionRefusedError:
            error_msg = "Connection refused"
            if self._config and self._config.mode == "builtin":
                error_msg += ". Built-in broker may not be running."
            else:
                error_msg += f". Please check if MQTT broker is running at {self.broker_host}:{self.broker_port}"
            logger.error(error_msg)
            self.is_connected = False
            self.recent_errors.append(
                {
                    "time": time.time(),
                    "type": "connection_refused",
                    "code": None,
                    "message": error_msg,
                }
            )
        except Exception as e:
            logger.error(f"Failed to connect: {e}", exc_info=True)
            self.is_connected = False
            self.recent_errors.append(
                {
                    "time": time.time(),
                    "type": "connection_error",
                    "code": None,
                    "message": str(e),
                }
            )
    
    def stop(self):
        """Stop MQTT client"""
        logger.info("Stopping MQTT client(s)...")
        try:
            # Stop all managed clients
            for client in self.clients or ([] if self.client is None else [self.client]):
                try:
                    client.loop_stop()
                    client.disconnect()
                except Exception as e:
                    logger.error(f"Error stopping MQTT client: {e}")
            self.clients = []
            self.client = None
            self.is_connected = False
            logger.info("MQTT client(s) stopped")
        except Exception as e:
            logger.error(f"Error stopping MQTT clients: {e}")

    def reload_and_reconnect(self):
        """Reload configuration from DB and reconnect to broker."""
        self.stop()
        self.start()


# Global MQTT service instance
mqtt_service = MQTTService()


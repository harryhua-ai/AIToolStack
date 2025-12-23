"""Built-in MQTT Broker service"""
import asyncio
import logging
import threading
import socket
import os
from pathlib import Path
from typing import Optional
from backend.config import settings, get_local_ip, BASE_DIR
from backend.services.mqtt_config_service import mqtt_config_service

logger = logging.getLogger(__name__)

# Password file path for aMQTT FileAuthPlugin
MQTT_PASSWORD_FILE = BASE_DIR / "data" / "mqtt_passwd"

try:
    from passlib.hash import sha512_crypt
    HAS_PASSLIB = True
except ImportError:
    HAS_PASSLIB = False
    logger.warning("[MQTT Broker] passlib not installed. Password hashing will use fallback method. "
                   "Please install passlib for proper password authentication: pip install passlib")


def create_password_file(username: str, password: str) -> str:
    """Create or update password file for aMQTT FileAuthPlugin.
    
    Format: username:sha512_crypt_hash (passlib format)
    Returns the path to the password file.
    """
    # Ensure data directory exists
    MQTT_PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate SHA-512 crypt hash using passlib (aMQTT FileAuthPlugin expects this format)
    if HAS_PASSLIB:
        password_hash = sha512_crypt.hash(password)
    else:
        # Fallback: use plain password (NOT SECURE, but allows basic functionality)
        # This should only happen if passlib is not installed
        logger.error("[MQTT Broker] passlib not available, using plain password (INSECURE). "
                    "Please install passlib: pip install passlib")
        password_hash = password
    
    # Read existing file to preserve other users (if any)
    existing_users = {}
    if MQTT_PASSWORD_FILE.exists():
        with open(MQTT_PASSWORD_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    user, _ = line.split(':', 1)
                    if user != username:  # Keep other users
                        existing_users[user] = line.split(':', 1)[1]
    
    # Write updated file
    with open(MQTT_PASSWORD_FILE, 'w', encoding='utf-8') as f:
        # Write the new/updated user
        f.write(f"{username}:{password_hash}\n")
        # Write other existing users
        for user, pwd_hash in existing_users.items():
            f.write(f"{user}:{pwd_hash}\n")
    
    logger.info(f"[MQTT Broker] Password file updated for user: {username}")
    return str(MQTT_PASSWORD_FILE)


def remove_password_file():
    """Remove password file if it exists."""
    if MQTT_PASSWORD_FILE.exists():
        try:
            MQTT_PASSWORD_FILE.unlink()
            logger.info("[MQTT Broker] Password file removed")
        except Exception as e:
            logger.warning(f"[MQTT Broker] Failed to remove password file: {e}")


class BuiltinMQTTBroker:
    """Built-in MQTT Broker (using aMQTT)"""
    
    def __init__(self):
        self.broker = None
        self.is_running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
    
    def _run_broker(self):
        """Run Broker in separate thread"""
        try:
            from amqtt.broker import Broker
            
            # Create new event loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            async def start_broker():
                # Load current MQTT config to drive built-in broker behaviour
                cfg = mqtt_config_service.load_config()

                # Build Broker configuration (aMQTT format)
                listeners = {
                    "default": {
                        "type": "tcp",
                        "bind": "0.0.0.0",
                        "port": cfg.builtin_tcp_port or settings.MQTT_BUILTIN_PORT,
                        "max-connections": cfg.builtin_max_connections,
                    }
                }

                # Optional TLS listener for MQTTS
                if cfg.builtin_tls_port and cfg.builtin_tls_enabled and cfg.builtin_tls_ca_cert_path:
                    listeners["tls"] = {
                        "type": "tcp",
                        "bind": "0.0.0.0",
                        "port": cfg.builtin_tls_port,
                        "max-connections": cfg.builtin_max_connections,
                        "ssl": True,
                        "cafile": cfg.builtin_tls_ca_cert_path,
                        "certfile": cfg.builtin_tls_client_cert_path or cfg.builtin_tls_ca_cert_path,
                        "keyfile": cfg.builtin_tls_client_key_path,
                    }

                # Configure authentication
                # NOTE: Newer aMQTT versions recommend configuring auth plugins
                # via the top-level "plugins" section, not via deprecated
                # "auth.plugins/password-file" keys.
                #
                # We keep "allow-anonymous" in auth section so that core
                # broker honours it, but actual username/password checking is
                # implemented by FileAuthPlugin configured in top-level plugins.
                auth_config = {
                    "allow-anonymous": cfg.builtin_allow_anonymous,
                }

                # Top-level plugins configuration for aMQTT
                # Example:
                # plugins:
                #   amqtt.plugins.authentication.FileAuthPlugin:
                #     password_file: /path/to/password_file
                #   amqtt.plugins.authentication.AnonymousAuthPlugin: {}
                plugins_dict = {}

                # If anonymous is disabled, only enable FileAuthPlugin
                if not cfg.builtin_allow_anonymous:
                    if cfg.builtin_username and cfg.builtin_password:
                        # Create/update password file with valid user
                        password_file_path = create_password_file(
                            cfg.builtin_username,
                            cfg.builtin_password
                        )
                        logger.info(f"[MQTT Broker] Authentication enabled with username: {cfg.builtin_username}")
                    else:
                        # No username/password configured: create password file
                        # with an invalid user so that all auth attempts fail.
                        logger.warning(
                            "[MQTT Broker] Anonymous access disabled but no username/password provided. "
                            "Creating password file with invalid user - all connections will be rejected "
                            "until username/password is configured."
                        )
                        MQTT_PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
                        with open(MQTT_PASSWORD_FILE, 'w', encoding='utf-8') as f:
                            f.write(
                                "__INVALID_USER__:"
                                "$6$rounds=5000$dummy$dummy_hash_that_will_never_match_any_password\n"
                            )
                    # In both cases above we use the same file path
                    plugins_dict["amqtt.plugins.authentication.FileAuthPlugin"] = {
                        "password_file": str(MQTT_PASSWORD_FILE)
                    }
                else:
                    # Anonymous allowed: enable AnonymousAuthPlugin
                    plugins_dict["amqtt.plugins.authentication.AnonymousAuthPlugin"] = {}
                    # If username/password are also configured, enable FileAuthPlugin
                    if cfg.builtin_username and cfg.builtin_password:
                        password_file_path = create_password_file(
                            cfg.builtin_username,
                            cfg.builtin_password
                        )
                        plugins_dict["amqtt.plugins.authentication.FileAuthPlugin"] = {
                            "password_file": str(password_file_path)
                        }
                        logger.info(
                            "[MQTT Broker] Both anonymous and authenticated access enabled "
                            f"(username: {cfg.builtin_username})"
                        )
                    else:
                        # No auth users when anonymous-only: remove any existing password file
                        remove_password_file()

                config = {
                    "listeners": listeners,
                    "sys_interval": 10,
                    "auth": auth_config,
                    "plugins": plugins_dict,
                    "keepalive-timeout": cfg.builtin_keepalive_timeout,
                }

                # Log authentication configuration for debugging
                password_file_set = bool(
                    plugins_dict.get("amqtt.plugins.authentication.FileAuthPlugin", {}).get("password_file")
                )
                auth_log_msg = (
                    "[MQTT Broker] Authentication config: "
                    f"allow-anonymous={cfg.builtin_allow_anonymous}, "
                    f"plugins={list(plugins_dict.keys())}, "
                    f"password_file={'set' if password_file_set else 'not set'}, "
                    f"auth={auth_config}, "
                    f"plugins_config={plugins_dict}"
                )
                logger.info(auth_log_msg)
                print(auth_log_msg)
                
                # Create and start Broker
                self.broker = Broker(config)
                await self.broker.start()
                self.is_running = True
                
                local_ip = get_local_ip()
                logger.info(f"[MQTT Broker] Built-in MQTT Broker started on port {listeners['default']['port']}")
                logger.info("[MQTT Broker] Protocol support: MQTT 3.1.1 only (MQTT 5.0 is not supported by aMQTT library)")
                if not cfg.builtin_allow_anonymous:
                    logger.info("[MQTT Broker] Anonymous access is DISABLED - clients must provide valid username/password")
                else:
                    logger.info("[MQTT Broker] Anonymous access is ENABLED - clients can connect without authentication")
                print(f"[MQTT Broker] Built-in MQTT Broker is ready at {local_ip}:{listeners['default']['port']}")
                print("[MQTT Broker] Note: Only MQTT 3.1.1 protocol is supported. Clients using MQTT 5.0 will be rejected.")
                
                # Keep running
                try:
                    while self.is_running:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("[MQTT Broker] Broker loop cancelled")
                finally:
                    self.is_running = False
                    if self.broker:
                        try:
                            await self.broker.shutdown()
                            logger.info("[MQTT Broker] Broker instance shut down")
                        except Exception as e:
                            logger.warning(f"[MQTT Broker] Error during broker shutdown: {e}")
            
            # Run async function - wrap in try/except to handle loop stopping
            try:
                self._loop.run_until_complete(start_broker())
            except RuntimeError as e:
                if "Event loop stopped" in str(e) or "Future completed" in str(e):
                    # This is expected when stopping the broker
                    logger.debug(f"[MQTT Broker] Event loop stopped: {e}")
                else:
                    raise
            
        except ImportError as e:
            logger.error(f"[MQTT Broker] aMQTT library not installed. Please install: pip install amqtt. Error: {e}")
            self.is_running = False
        except Exception as e:
            logger.error(f"[MQTT Broker] Failed to start built-in broker: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_running = False
    
    def start(self):
        """Start built-in MQTT Broker (synchronous method)"""
        if self.is_running:
            logger.warning("[MQTT Broker] Broker is already running")
            return
        
        try:
            # Start Broker in separate thread
            self._thread = threading.Thread(target=self._run_broker, daemon=True)
            self._thread.start()
            
            # Wait a bit to ensure Broker starts
            import time
            time.sleep(0.5)
            
            if not self.is_running:
                raise RuntimeError("Failed to start built-in MQTT Broker")
                
        except Exception as e:
            logger.error(f"[MQTT Broker] Error starting broker thread: {e}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop built-in MQTT Broker"""
        if not self.is_running and not self._thread:
            return
        
        logger.info("[MQTT Broker] Stopping broker...")
        import time
        
        try:
            # Set flag to stop the broker loop (this will exit the while loop in start_broker)
            self.is_running = False
            
            # Give the loop a moment to exit naturally
            time.sleep(0.5)
            
            # Stop broker instance if it exists
            if self.broker and self._loop:
                try:
                    # Schedule broker shutdown in the event loop
                    if self._loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            self.broker.shutdown(), 
                            self._loop
                        )
                        # Wait for shutdown to complete (with timeout)
                        try:
                            future.result(timeout=2)
                        except Exception as e:
                            logger.warning(f"[MQTT Broker] Broker shutdown timeout or error: {e}")
                except Exception as e:
                    logger.warning(f"[MQTT Broker] Error scheduling broker shutdown: {e}")
            
            # Stop event loop gracefully - only if it's still running
            if self._loop and self._loop.is_running():
                try:
                    # Cancel all pending tasks first
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    
                    # Stop the loop
                    self._loop.call_soon_threadsafe(self._loop.stop)
                    # Give it a moment to process the stop
                    time.sleep(0.2)
                except Exception as e:
                    logger.warning(f"[MQTT Broker] Error stopping event loop: {e}")
            
            # Wait for thread to finish
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    logger.warning("[MQTT Broker] Thread did not stop within timeout, forcing cleanup")
            
            # Clean up
            self.broker = None
            self._loop = None
            self._thread = None
            
            # Additional wait to ensure port is released
            time.sleep(0.5)
            
            logger.info("[MQTT Broker] Built-in MQTT Broker stopped")
        except Exception as e:
            logger.error(f"[MQTT Broker] Error stopping broker: {e}")
            # Force cleanup on error
            self.broker = None
            self._loop = None
            self._thread = None
            self.is_running = False

    def restart(self):
        """Restart built-in MQTT Broker with latest configuration."""
        logger.info("[MQTT Broker] Restarting broker...")
        was_running = self.is_running
        self.stop()
        # Wait longer for broker to fully stop and release port
        import time
        time.sleep(2.5)
        if was_running:
            self.start()
            # Wait a bit for broker to start (start is async)
            time.sleep(2)
    
    def get_broker_address(self) -> str:
        """Get Broker address"""
        local_ip = get_local_ip()
        return f"{local_ip}:{settings.MQTT_BUILTIN_PORT}"


# Global built-in Broker instance
builtin_mqtt_broker = BuiltinMQTTBroker()

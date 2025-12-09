"""配置文件"""
import socket
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"


def get_local_ip() -> str:
    """获取本机 IP 地址"""
    try:
        # 连接到一个远程地址（不需要实际连接）
        # 这样可以获取本机用于连接外网的 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # 不需要真正连接，只是获取本机 IP
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            # 如果失败，尝试获取 localhost
            ip = socket.gethostbyname(socket.gethostname())
        finally:
            s.close()
        return ip
    except Exception:
        # 如果都失败，返回 localhost
        return "127.0.0.1"


class Settings(BaseSettings):
    """应用配置"""
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # MQTT 配置
    MQTT_ENABLED: bool = True  # 是否启用 MQTT 服务
    MQTT_USE_BUILTIN_BROKER: bool = True  # 是否使用内置 Broker（默认使用内置）
    MQTT_BROKER: str = ""  # 外部 Broker 地址（当 MQTT_USE_BUILTIN_BROKER=False 时使用，为空则自动使用本机 IP）
    MQTT_PORT: int = 1883  # MQTT 端口
    MQTT_BUILTIN_PORT: int = 1883  # 内置 Broker 端口
    MQTT_USERNAME: str = ""  # 外部 Broker 认证（内置 Broker 暂不支持认证）
    MQTT_PASSWORD: str = ""
    MQTT_UPLOAD_TOPIC: str = "annotator/upload/+"
    MQTT_RESPONSE_TOPIC_PREFIX: str = "annotator/response"
    MQTT_QOS: int = 1
    
    # 数据库配置
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/annotator.db"
    
    # 文件存储配置
    DATASETS_ROOT: Path = DATASETS_DIR
    MAX_IMAGE_SIZE_MB: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# 如果 MQTT_BROKER 为空，使用本机 IP
if not settings.MQTT_BROKER:
    settings.MQTT_BROKER = get_local_ip()

# 确保必要目录存在
settings.DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


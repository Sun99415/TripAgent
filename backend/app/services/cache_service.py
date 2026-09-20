"""Redis 结果缓存服务(可选,fail-open 降级)

本服务用于缓存最终旅行计划结果,命中时可直接返回,跳过昂贵的多智能体调用。

设计原则(fail-open):
- 未安装 redis 库 / 未配置 REDIS_URL / Redis 连接失败时,自动降级为"无缓存",
  不影响主流程。
"""

import json
from typing import Any, Optional

from loguru import logger

from ..config import get_settings

try:
    import redis
except ImportError:  # redis 未安装时降级
    redis = None


class TripCacheService:
    """旅行计划结果缓存(键值对存 JSON)"""

    def __init__(self) -> None:
        self._client = None
        self._enabled = False
        self._ttl = 3600
        self._init_client()

    def _init_client(self) -> None:
        if redis is None:
            logger.warning("redis 库未安装(缺少依赖),结果缓存已关闭")
            return

        settings = get_settings()
        if not settings.redis_url:
            logger.info("未配置 REDIS_URL,结果缓存已关闭")
            return

        try:
            self._client = redis.Redis.from_url(
                settings.redis_url,
                socket_connect_timeout=1,
                socket_timeout=1,
                decode_responses=True,
                # 显式用 RESP2 协议,兼容老版本 Redis(<6.0 不支持 HELLO/RESP3)
                protocol=2,
            )
            self._client.ping()
            self._enabled = True
            self._ttl = settings.redis_ttl
            logger.info("Redis 结果缓存已启用: {}", settings.redis_url)
        except Exception as e:  # noqa: BLE001 - 连接失败降级为关闭
            logger.warning("Redis 连接失败({}),结果缓存降级为关闭", e)
            self._client = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    def get_json(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 读取失败({}),本次跳过缓存", e)
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            self._client.set(
                key,
                json.dumps(value, ensure_ascii=False),
                ex=ttl if ttl is not None else self._ttl,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 写入失败({})", e)


# 全局缓存实例(单例)
_cache_instance: Optional[TripCacheService] = None


def get_cache() -> TripCacheService:
    """获取结果缓存实例(单例)"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TripCacheService()
    return _cache_instance
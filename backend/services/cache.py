import time
from typing import Any, Dict
from backend.config import settings

class SimpleCache:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = settings.CACHE_TTL

    def get(self, key: str) -> Any:
        entry = self._cache.get(key)
        if entry and (time.time() - entry["timestamp"] < self._ttl):
            return entry["value"]
        return None

    def set(self, key: str, value: Any):
        self._cache[key] = {"value": value, "timestamp": time.time()}

    def clear(self):
        self._cache.clear()

cache = SimpleCache()
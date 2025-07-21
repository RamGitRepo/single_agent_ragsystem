import os, redis, json, uuid
from datetime import timedelta

r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_KEY", None),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        decode_responses=True
    )

TTL_HOURS = 24      # how long to keep a convo

def _key(cid: str) -> str:
    return f"chat:{cid}"

def append(cid: str, role: str, text: str):
    r.rpush(_key(cid), json.dumps({"role": role, "text": text}))
    r.expire(_key(cid), timedelta(hours=TTL_HOURS))

def fetch(cid: str, last_n: int = 6):
    items = r.lrange(_key(cid), -2*last_n, -1)  # each Q/A is two items
    print(f"Fetching last {last_n} items for conversation {cid}: {items}")
    return [json.loads(x) for x in items]

def get(key: str):
    value = r.get(key)
    return json.loads(value) if value else None

def set(key: str, value: str):
    r.set(key, json.dumps(value), ex=TTL_HOURS * 3600)

def new_conversation_id() -> str:
    return str(uuid.uuid4())
import redis, hashlib, json, os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
r = redis.Redis(host=os.getenv("REDIS_HOST"), port=6379, password=os.getenv("REDIS_KEY"))

def _make_key(question: str, doc_ids: list[int]) -> str:
    payload = question + "|" + "|".join(map(str, sorted(doc_ids)))
    return hashlib.sha256(payload.encode()).hexdigest()

def get_cached_answer(question, doc_ids):
    key = _make_key(question, doc_ids)
    print(f"Looking for cached answer with key: {key}")
    print(f"Redis host: {os.getenv('REDIS_HOST')}")
    val = r.get(key)
    return json.loads(val) if val else None

def set_cached_answer(question, doc_ids, answer, ttl=3600):
    key = _make_key(question, doc_ids)
    r.set(key, json.dumps(answer), ex=ttl)

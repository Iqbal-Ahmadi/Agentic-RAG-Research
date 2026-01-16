import re

_INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"reveal (the )?system prompt",
    r"developer message",
    r"system prompt",
]

def validate_user_question(q: str, max_len: int) -> str:
    q = (q or "").strip()
    if len(q) < 5:
        raise ValueError("Query is too short.")
    if len(q) > max_len:
        raise ValueError("Query is too long.")
    low = q.lower()
    for p in _INJECTION_PATTERNS:
        if re.search(p, low):
            raise ValueError("Potential prompt injection detected.")
    return q

def validate_retrieval_params(top_k: int, max_k: int = 10) -> int:
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if top_k > max_k:
        return max_k
    return top_k

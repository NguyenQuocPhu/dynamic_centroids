

from typing import List
from functools import wraps

def count_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

def count_batch_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        batch = kwargs.get('conversations')
        if batch is None and len(args) > 1:
            batch = args[1]
        
        if isinstance(batch, list):
            wrapper.calls += len(batch)
        else:
            wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper
"""Low-overhead per-invocation measurements; inclusive times must not be added."""
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import json
import time

_CURRENT = ContextVar('asr2rpp_performance', default=None)


def record(name, elapsed):
    state = _CURRENT.get()
    if state is not None:
        item = state['timings'].setdefault(name, {'calls':0,'seconds':0.0})
        item['calls'] += 1
        item['seconds'] += elapsed


def timed(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            label = name(*args, **kwargs) if callable(name) else name
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                record(label, time.perf_counter()-started)
        return call
    return decorate


def report_directory(path):
    state = _CURRENT.get()
    if state is not None:
        state['reports'].add(Path(path))


def profiled(function):
    @wraps(function)
    def call(*args, **kwargs):
        if _CURRENT.get() is not None:
            return function(*args, **kwargs)
        state={'timings':{},'reports':set()}
        token=_CURRENT.set(state)
        started=time.perf_counter()
        status='failed'
        try:
            value=function(*args, **kwargs)
            status='returned'
            return value
        finally:
            payload={'schema':1,'scope':'pipeline invocation (queue totals shared)',
                     'inclusive_times':True,'total_seconds':time.perf_counter()-started,
                     'status':status,'timings':state['timings']}
            _CURRENT.reset(token)
            for path in state['reports']:
                if path.is_dir():
                    try:
                        (path/'performance.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
                    except OSError:
                        pass  # Advisory telemetry must not mask success/cancellation/errors.
    return call

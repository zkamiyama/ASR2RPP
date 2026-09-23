"""Join timed text without adding spaces inside subword tokens."""
def join_timed(previous: str, current: str, granularity: str) -> str:
    current = current.replace('\u2581', ' ')
    if granularity == 'token':
        return previous + current
    if previous and current and previous[-1].isascii() and previous[-1].isalnum() and current[0].isascii() and current[0].isalnum():
        return previous + ' ' + current
    return previous + current

"""One terminal event per item per attempt; completed output never becomes pending."""
from dataclasses import dataclass, field

RETRYABLE = frozenset({'waiting', 'failed', 'stopped'})
TERMINAL = frozenset({'done', 'failed', 'stopped'})
_STATUS = {'実行中': 'running', '完了': 'done', '失敗': 'failed', '中断': 'stopped'}


def canonical_status(value):
    return _STATUS.get(value, value if value in TERMINAL or value == 'waiting' else 'running')


@dataclass
class RunLedger:
    indices: tuple
    records: dict = field(init=False)

    def __post_init__(self):
        if len(set(self.indices)) != len(self.indices):
            raise ValueError('Queue indices must be unique within an attempt')
        self.records = {i: ('waiting', '') for i in self.indices}

    def accept(self, index, status, detail=''):
        if index not in self.records or self.records[index][0] in TERMINAL:
            return None
        status = canonical_status(status)
        self.records[index] = (status, str(detail or ''))
        return self.records[index]

    @property
    def pending(self):
        return [i for i, (state, _) in self.records.items() if state not in TERMINAL]

    @property
    def completed(self):
        return len(self.records) - len(self.pending)

import json
import threading
from unittest.mock import patch
import pytest
from scripts.benchmark_api import run_trial


class Connection:
    instances = []
    status = 200

    def __init__(self, *args, **kwargs):
        self.owner = threading.get_ident()
        self.calls = 0
        self.closed = False
        self.instances.append(self)

    def request(self, method, path, body, headers):
        assert threading.get_ident() == self.owner
        self.calls += 1

    def getresponse(self):
        return self

    def read(self):
        return json.dumps({"recommendations": []}).encode()

    def close(self):
        self.closed = True


def test_connections_reused_and_closed():
    Connection.instances = []
    with patch('scripts.benchmark_api.http.client.HTTPConnection', Connection):
        result = run_trial('http://localhost:8000/recommend', ['u', 'v'], 100, 4, 10)
    assert 1 <= len(Connection.instances) <= 4
    assert sum(c.calls for c in Connection.instances) == 110
    assert all(c.closed for c in Connection.instances)
    assert result['requests'] == 100


def test_http_failure_not_counted_as_success():
    Connection.instances = []
    with patch('scripts.benchmark_api.http.client.HTTPConnection', Connection), patch.object(Connection, 'status', 500):
        with pytest.raises(RuntimeError, match='HTTP 500'):
            run_trial('http://localhost/recommend', ['u'], 1, 1, 0)
    assert all(c.closed for c in Connection.instances)

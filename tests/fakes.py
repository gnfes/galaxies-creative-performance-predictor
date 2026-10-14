"""Simulador da YouTube Data API para os testes (sem rede nem cota)."""

import httplib2
from googleapiclient.errors import HttpError

from creative_predictor.collect.youtube import YouTubeClient


class FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeResource:
    def __init__(self, handler):
        self._handler = handler

    def list(self, **params):
        return FakeRequest(self._handler(params))


class FakeService:
    """Imita `build("youtube", "v3")`: cada recurso responde por uma função."""

    def __init__(self, **handlers):
        self.handlers = handlers
        self.calls = []

    def __getattr__(self, resource):
        handler = self.handlers[resource]

        def logged(params):
            self.calls.append((resource, params))
            return handler(params)

        return lambda: FakeResource(logged)


def make_client(tmp_path, **handlers):
    service = FakeService(**handlers)
    return YouTubeClient(service=service, cache_dir=tmp_path), service


def not_found():
    return HttpError(httplib2.Response({"status": 404}), b"playlistNotFound")

import pytest
import requests

from safepath.data.download import ArcGISError, get_json, objectid_windows


def test_windows_cover_range_without_overlap():
    windows = objectid_windows(1, 4500, 2000)
    assert windows == [(0, 2000), (2000, 4000), (4000, 4500)]
    covered = [i for lo, hi in windows for i in range(lo + 1, hi + 1)]
    assert covered == list(range(1, 4501))


def test_windows_exact_multiple():
    assert objectid_windows(1, 4000, 2000) == [(0, 2000), (2000, 4000)]


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, params, timeout):
        self.calls += 1
        return self.responses.pop(0)


def test_get_json_retries_http_and_arcgis_errors():
    session = FakeSession(
        [
            FakeResponse({}, status=503),
            FakeResponse({"error": {"code": 500, "message": "busy"}}),
            FakeResponse({"count": 7}),
        ]
    )
    body = get_json(session, "u", {}, timeout=1, max_retries=3, backoff=0)
    assert body == {"count": 7}
    assert session.calls == 3


def test_get_json_gives_up_after_max_retries():
    session = FakeSession([FakeResponse({"error": {"code": 500}})] * 3)
    with pytest.raises(ArcGISError):
        get_json(session, "u", {}, timeout=1, max_retries=2, backoff=0)
    assert session.calls == 3

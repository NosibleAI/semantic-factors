"""Price-provider failures stay inside the package exception contract."""
import datetime as dt
import json
import traceback

import pytest
import requests

from semantic_factors import returns as return_data
from semantic_factors.errors import CredentialError, ReturnsUnavailableError


class SequenceSession:
    """Return prepared responses or exceptions in call order."""

    def __init__(self, outcomes):
        """Retain a mutable queue and a call counter."""
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, **kwargs):
        """Return or raise the next prepared outcome."""
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def response(status: int, payload, retry_after: str = "") -> requests.Response:
    """Build a requests response without network access."""
    prepared = requests.Response()
    prepared.status_code = status
    prepared._content = (
        payload if isinstance(payload, bytes) else json.dumps(payload).encode(encoding="utf-8")
    )
    prepared.url = "https://eodhd.example/api/eod/TEST.US"
    if retry_after:
        prepared.headers["Retry-After"] = retry_after
    return prepared


def call_download(session: SequenceSession):
    """Invoke one deterministic ticker request."""
    return return_data.download_closes(
        session=session,
        ticker="TEST.US",
        start=dt.date(2024, 1, 1),
        stop=dt.date(2024, 1, 2),
        api_key="test-key",
    )


def test_eodhd_transient_is_retried_and_then_parsed(monkeypatch) -> None:
    """Recover from a capacity response and sort a complete valid price frame."""
    session = SequenceSession(
        [
            response(503, {"error": "busy"}, retry_after="0.1"),
            response(
                200,
                [
                    {"date": "2024-01-02", "adjusted_close": 11},
                    {"date": "2024-01-01", "adjusted_close": 10},
                ],
            ),
        ]
    )
    sleeps = []
    monkeypatch.setattr(return_data.random, "uniform", lambda low, high: 0.0)
    monkeypatch.setattr(return_data.time, "sleep", sleeps.append)

    frame = call_download(session=session)

    assert session.calls == 2
    assert sleeps == [0.1]
    assert frame["date"].to_list() == [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
    assert frame["TEST.US"].to_list() == [10.0, 11.0]


def test_eodhd_transport_exhaustion_is_a_package_exception(monkeypatch) -> None:
    """Never leak requests.ConnectionError after bounded retries are exhausted."""
    session = SequenceSession(
        [requests.ConnectionError("offline") for _ in range(return_data.EODHD_MAX_ATTEMPTS)]
    )
    monkeypatch.setattr(return_data.time, "sleep", lambda seconds: None)

    with pytest.raises(ReturnsUnavailableError, match="after 4 attempts"):
        call_download(session=session)

    assert session.calls == return_data.EODHD_MAX_ATTEMPTS


def test_eodhd_transport_exception_cannot_leak_a_secret(monkeypatch) -> None:
    """Keep credential-bearing request URLs out of the package's public error message."""
    secret = "SECRET-TRANSPORT"
    secret_url = f"https://eodhd.example/eod?api_token={secret}"
    session = SequenceSession(
        [
            requests.ConnectionError(f"failed URL {secret_url}")
            for _ in range(return_data.EODHD_MAX_ATTEMPTS)
        ]
    )
    monkeypatch.setattr(return_data.time, "sleep", lambda seconds: None)

    with pytest.raises(ReturnsUnavailableError) as raised:
        call_download(session=session)

    rendered = "".join(
        traceback.format_exception(raised.type, raised.value, raised.tb, chain=True)
    )
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
    assert secret not in rendered
    assert secret_url not in rendered


def test_eodhd_transient_status_exhaustion_is_a_package_exception(monkeypatch) -> None:
    """Normalize the last retryable HTTP response after the bounded retry budget."""
    session = SequenceSession(
        [
            response(503, {"error": "busy"})
            for _ in range(return_data.EODHD_MAX_ATTEMPTS)
        ]
    )
    monkeypatch.setattr(return_data.time, "sleep", lambda seconds: None)

    with pytest.raises(ReturnsUnavailableError, match="HTTP 503"):
        call_download(session=session)

    assert session.calls == return_data.EODHD_MAX_ATTEMPTS


@pytest.mark.parametrize(
    "prepared",
    [
        response(200, b"not-json"),
        response(200, {"error": "quota"}),
        response(200, [{"adjusted_close": 10.0}]),
        response(200, [{"date": "2024-01-01", "adjusted_close": float("nan")}]),
        response(
            200,
            [
                {"date": "2024-01-01", "adjusted_close": 10.0},
                {"date": "2024-01-01", "adjusted_close": 11.0},
            ],
        ),
    ],
)
def test_eodhd_schema_failures_are_package_exceptions(prepared) -> None:
    """Normalize malformed JSON and rows before Polars or builtins can leak through."""
    with pytest.raises(ReturnsUnavailableError):
        call_download(session=SequenceSession([prepared]))


def test_eodhd_http_failures_are_normalized_without_retrying_permanent_errors() -> None:
    """Keep credential and permanent provider failures precise and bounded."""
    unauthorized = SequenceSession([response(401, {"error": "invalid token"})])
    with pytest.raises(CredentialError):
        call_download(session=unauthorized)
    assert unauthorized.calls == 1

    bad_request = SequenceSession([response(400, {"error": "bad symbol"})])
    with pytest.raises(ReturnsUnavailableError, match="HTTP 400"):
        call_download(session=bad_request)
    assert bad_request.calls == 1


def test_eodhd_http_body_cannot_leak_a_secret() -> None:
    """Never copy an untrusted provider body into a caller-visible package exception."""
    secret = "api_token=SECRET-BODY"
    bad_request = SequenceSession([response(400, {"error": secret})])

    with pytest.raises(ReturnsUnavailableError) as raised:
        call_download(session=bad_request)

    rendered = "".join(
        traceback.format_exception(raised.type, raised.value, raised.tb, chain=True)
    )
    assert secret not in rendered

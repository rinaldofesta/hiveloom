import signal
import threading
import time
from urllib.error import URLError

import pytest

from hiveloom.deadline import DeadlineExceeded, absolute_deadline, remaining_seconds
from hiveloom.models.openai_compat import OpenAICompatProvider


@pytest.mark.parametrize("seconds", [0, -1, 241, float("inf"), float("nan")])
def test_invalid_budget_fails_closed(seconds):
    with pytest.raises(ValueError), absolute_deadline(seconds):
        pytest.fail("invalid deadline entered")


def test_retries_share_one_deadline(monkeypatch):
    timeouts = []

    def unavailable(request, *, timeout):
        timeouts.append(timeout)
        time.sleep(0.04)
        raise URLError("synthetic transient failure")

    monkeypatch.setattr("hiveloom.models.openai_compat.urlrequest.urlopen", unavailable)
    provider = OpenAICompatProvider(
        "https://invalid.example", sleep=lambda _: time.sleep(0.02)
    )
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded), absolute_deadline(0.15):
        provider._post("/chat/completions", {})
    assert 2 <= len(timeouts) <= 3
    assert all(0 < timeout <= 0.15 for timeout in timeouts)
    assert timeouts[-1] < timeouts[0]
    assert time.monotonic() - started < 0.6


def test_retry_backoff_is_interruptible(monkeypatch):
    def unavailable(request, *, timeout):
        raise URLError("synthetic transient failure")

    monkeypatch.setattr("hiveloom.models.openai_compat.urlrequest.urlopen", unavailable)
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded), absolute_deadline(0.05):
        OpenAICompatProvider("https://invalid.example")._post("/test", {})
    assert time.monotonic() - started < 0.5


def test_expired_nested_scope_does_not_rearm_during_unwind():
    with pytest.raises(DeadlineExceeded), absolute_deadline(0.02):
        with absolute_deadline(1):
            time.sleep(1)
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
    time.sleep(0.01)


def test_nested_budget_cannot_extend_parent():
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded), absolute_deadline(0.1):
        time.sleep(0.03)
        with absolute_deadline(240):
            time.sleep(1)
    assert time.monotonic() - started < 0.5


def test_timer_is_restored_after_success_and_failure():
    previous = signal.getsignal(signal.SIGALRM)
    with absolute_deadline(1):
        assert remaining_seconds(120) <= 1
    with pytest.raises(DeadlineExceeded), absolute_deadline(0.02):
        time.sleep(1)
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert remaining_seconds(120) == 120


def test_non_main_thread_refuses_unenforced_deadline():
    failures = []

    def attempt():
        try:
            with absolute_deadline(1):
                pass
        except RuntimeError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join()
    assert failures == ["absolute_deadline needs a dedicated POSIX main thread"]


def test_preexisting_alarm_is_not_displaced():
    signal.setitimer(signal.ITIMER_REAL, 10)
    try:
        with pytest.raises(RuntimeError), absolute_deadline(1):
            pytest.fail("displaced another alarm")
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 9
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

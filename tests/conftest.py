import time
import pytest

test_durations = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    test_durations.append((item.nodeid, duration))


def pytest_sessionfinish(session, exitstatus):
    print("\nTest durations:")
    total_time = sum(d for _, d in test_durations)
    # for name, duration in test_durations:
    #    print(f"{name}: {duration:.4f} seconds")
    if test_durations:
        avg = total_time / len(test_durations)
        print(f"\nAverage test duration: {avg:.4f} seconds")

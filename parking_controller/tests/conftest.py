import pickle
from pathlib import Path

import pytest
from test_park import metric_dict


def pytest_configure(config: pytest.Config):
    assert isinstance(config, pytest.Config)
    config._record_metric_storage = {}  # type: ignore


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    print(session, exitstatus)
    assert isinstance(session, pytest.Session)
    # Access the stored metrics and write them to a file at the end of the test
    record_metric: metric_dict = session.config._record_metric_storage  # type: ignore
    print("metrics:")
    for x, y in record_metric.items():
        print(x)
        print(y)
        print()

    with open(Path(__file__).parent / "record_metric.pkl", "wb") as file:
        pickle.dump(record_metric, file)

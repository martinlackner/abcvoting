import io
import logging
import pytest

from abcvoting.output import Output, VERBOSITY_TO_NAME, DETAILS, INFO


@pytest.mark.parametrize("verbosity", VERBOSITY_TO_NAME.keys())
def test_verbosity(capfd, verbosity):
    output = Output(verbosity=verbosity)
    output.debug2("debug2")
    output.debug("debug")
    output.details("details")
    output.info("info")
    output.warning("warning")
    output.error("error")
    output.critical("critical")

    stdout = capfd.readouterr().out
    for verbosity_value, verbosity_name in VERBOSITY_TO_NAME.items():
        if verbosity_value >= verbosity:
            assert verbosity_name.lower() in stdout
        else:
            assert verbosity_name.lower() not in stdout


def test_verbosity2(capfd):
    output = Output(verbosity=INFO)
    output.details("details")
    output.info("info")

    stdout = capfd.readouterr().out

    assert "info\n" in stdout
    assert "details\n" not in stdout

    output.set_verbosity(DETAILS)
    output.details("details")
    output.info("info")

    stdout = capfd.readouterr().out

    assert "info\n" in stdout
    assert "details\n" in stdout


@pytest.mark.parametrize("verbosity", [INFO, DETAILS])
def test_logger(capfd, verbosity):
    logger = logging.getLogger("testoutput")
    logger.setLevel(logging.DEBUG)

    logger_output = io.StringIO("test")
    handler = logging.StreamHandler(stream=logger_output)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    output = Output(verbosity=verbosity, logger=logger)
    output.info("info")
    output.debug2("debug2")
    output.debug("debug")
    output.details("details")

    handler.flush()

    stdout = capfd.readouterr().out
    logger_output_str = logger_output.getvalue()

    assert "info\n" in stdout
    assert "debug2\n" not in stdout
    if verbosity <= DETAILS:
        assert "details\n" in stdout

    # always printed, independent of verbosity, determined by logger's level
    assert "info\n" in logger_output_str
    assert "details\n" in logger_output_str
    assert "debug2\n" in logger_output_str

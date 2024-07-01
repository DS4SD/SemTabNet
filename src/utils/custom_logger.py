"""A customized logger."""
import logging
from pathlib import Path
from typing import Optional

# DEBUG   Detailed information, typically of interest only when diagnosing problems.
#
# INFO    Confirmation that things are working as expected.
#
# WARNING An indication that something unexpected happened,
# or indicative of some problem in the near future (e.g. ‘disk space low’).
# The software is still working as expected.
#
# ERROR   Due to a more serious problem, the software has not been able
# to perform some function.
#
# CRITICAL A serious error, indicating that the program
# itself may be unable to continue running.


def get_custom_logger(
    filepath: str, level: Optional[str] = None, logfile: Optional[Path] = None
) -> logging.Logger:
    """Return a logger that displayes informations including package name.

    Args:
        filepath: The filepath of the caller.
        level: The logger level.

    Returns:
        A logger.
    """
    logger = logging.getLogger(filepath.split("cps-nlp-llm")[-1])

    if not level:
        level = logging.INFO  # type: ignore
    else:
        level = logging.getLevelName(level.upper())  # type: ignore

    logger.setLevel(level)  # type: ignore
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)8s] [%(filename)20s:%(lineno)4s] --- %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    if not logfile:
        logfile = Path("logs/log.log")
        if not logfile.exists():
            Path.mkdir(logfile.parent, parents=True)
            open(logfile, "a").close()
    else:
        if not logfile.parent.exists():
            Path.mkdir(logfile.parent, parents =True)
        if not logfile.exists():
            open(logfile, "a").close()

    file_handler = logging.FileHandler(filename=logfile, encoding="utf8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

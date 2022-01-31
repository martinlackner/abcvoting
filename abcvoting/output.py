"""
Print messages to terminal depending on a given verbosity level.

Similar to the Python logging module. Also meant to be used as Singleton.

.. important::

    The verbosity levels are:
    - CRITICAL
    - ERROR
    - WARNING
    - INFO
    - DETAILS
    - DEBUG
    - DEBUG2

"""

import logging

# should match the values defined in the logging module!
CRITICAL = 50
ERROR = 40
WARNING = 30
INFO = 20
DETAILS = 15
DEBUG = 10
DEBUG2 = 5

VERBOSITY_TO_NAME = {
    CRITICAL: "CRITICAL",
    ERROR: "ERROR",
    WARNING: "WARNING",
    INFO: "INFO",
    DETAILS: "DETAILS",
    DEBUG: "DEBUG",
    DEBUG2: "DEBUG2",
}


class Output:
    """
    Handling the output based on the current verbosity level.

    This is inspired by the class ``logging.Logger()``. A verbosity level is stored,
    only messages printed with methods with higher importance will be printed.

    Parameters
    ----------

        verbosity : int
            Verbosity level.

             Minimum level of importance of messages to be printed, as defined by
             constants in this module

        logger : logging.Logger, optional
             Optional logger.

             Can be used to send messages also to a log file or elsewhere, log level is separate
             from verbosity.
    """

    def __init__(self, verbosity=WARNING, logger=None):
        """
        Initialize the unique Output object.

        At the moment only one instance will created: in the global scope of this module. An
        application might use the `setup()` method to set the verbosity and a logger.
        """
        self.verbosity = verbosity
        self.logger = logger

    def set_verbosity(self, verbosity):
        """
        Set verbosity level.

        Parameters
        ----------

            verbosity : int
                Verbosity level.
        """
        self.verbosity = verbosity

    def _print(self, verbosity, msg):
        if verbosity >= self.verbosity:
            print(msg)

        if self.logger:
            self.logger.log(verbosity if verbosity not in (DETAILS, DEBUG2) else DEBUG, msg)

    def debug2(self, msg):
        """
        Print a message with verbosity level DEBUG2.

        Parameters
        ----------
            msg : str
                The message.
        """
        self._print(DEBUG2, msg)

    def debug(self, msg):
        """
        Print a message with verbosity level DEBUG.

        Parameters
        ----------
            msg : str
                The message.
        """
        # this is the old verbose >= 3
        self._print(DEBUG, msg)

    def details(self, msg):
        """
        Print a message with verbosity level DETAILS.

        Parameters
        ----------
            msg : str
                The message.
        """
        # this is the old verbose >= 2
        self._print(DETAILS, msg)

    def info(self, msg):
        """
        Print a message with verbosity level INFO.

        Parameters
        ----------
            msg : str
                The message.
        """
        # this is the old verbose >= 1
        self._print(INFO, msg)

    def warning(self, msg):
        """
        Print a message with verbosity level WARNING.

        Parameters
        ----------
            msg : str
                The message.
        """
        # this is the old verbose >= 0
        self._print(WARNING, msg)

    def error(self, msg):
        """
        Print a message with verbosity level ERROR.

        Parameters
        ----------
            msg : str
                The message.
        """
        # to print errors, probably not used atm
        self._print(ERROR, msg)

    def critical(self, msg):
        """
        Print a message with verbosity level CRITICAL.

        Parameters
        ----------
            msg : str
                The message.
        """
        # just for consistency with the logging module
        self._print(CRITICAL, msg)


output = Output()

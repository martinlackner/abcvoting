"""
Print messages to terminal depending on a given verbosity level.

Similar to the Python logging module. Also meant to be used as Singleton.

The verbosity levels are:

- CRITICAL
- ERROR
- WARNING
- INFO
- DETAILS
- DEBUG
- DEBUG2

The default verbosity is `WARNING`.

"""

import textwrap

# should match the values defined in the logging module!
CRITICAL = 50
ERROR = 40
WARNING = 30
INFO = 20
DETAILS = 15
DEBUG = 10
DEBUG2 = 5

DEFAULT = WARNING

WIDTH = 70  # default line width for output

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

    def __init__(self, verbosity=DEFAULT, logger=None):
        """
        Initialize the unique Output object.

        At the moment only one instance will created: in the global scope of this module. An
        application might use the `setup()` method to set the verbosity and a logger.
        """
        self.verbosity = verbosity
        self.logger = logger

    def set_verbosity(self, verbosity=DEFAULT):
        """
        Set verbosity level.

        Parameters
        ----------
            verbosity : int
                Verbosity level.
        """
        self.verbosity = verbosity

    def _print(self, verbosity, msg, wrap, indent):
        if verbosity >= self.verbosity:
            if wrap:
                input_msg = msg.split("\n")
                msg = "\n".join(
                    [
                        textwrap.fill(
                            line,
                            width=WIDTH,
                            break_long_words=False,
                            initial_indent=indent,
                            subsequent_indent=indent,
                        )
                        for line in input_msg
                    ]
                )
            print(msg)

        if self.logger:
            self.logger.log(verbosity if verbosity not in (DETAILS, DEBUG2) else DEBUG, msg)

    def debug2(self, msg, wrap=True, indent=""):
        """
        Print a message with verbosity level DEBUG2.

        Parameters
        ----------
            msg : str
                The message.

            wrap : bool, optional
                Wrap the message at 99 characters (if too long).

            indent : str, optional
                Indent each line with the this string.
        """
        self._print(DEBUG2, msg, wrap, indent)

    def debug(self, msg, wrap=True, indent=""):
        """
        Print a message with verbosity level DEBUG.

        Parameters
        ----------
            msg : str
                The message.

            wrap : bool, optional
                Wrap the message at 99 characters (if too long).

            indent : str, optional
                Indent each line with the this string.
        """
        # this is the old verbose >= 3
        self._print(DEBUG, msg, wrap, indent)

    def details(self, msg, wrap=True, indent=""):
        """
        Print a message with verbosity level DETAILS.

        Parameters
        ----------
            msg : str
                The message.

            wrap : bool, optional
                Wrap the message at 99 characters (if too long).

            indent : str, optional
                Indent each line with the this string.
        """
        # this is the old verbose >= 2
        self._print(DETAILS, msg, wrap, indent)

    def info(self, msg, wrap=True, indent=""):
        """
        Print a message with verbosity level INFO.

        Parameters
        ----------
            msg : str
                The message.

            wrap : bool, optional
                Wrap the message at 99 characters (if too long).

            indent : str, optional
                Indent each line with the this string.
        """
        # this is the old verbose >= 1
        self._print(INFO, msg, wrap, indent)

    def warning(self, msg, wrap=True, indent=""):
        """
        Print a message with verbosity level WARNING.

        Parameters
        ----------
            msg : str
                The message.

            wrap : bool, optional
                Wrap the message at 99 characters (if too long).

            indent : str, optional
                Indent each line with the this string.
        """
        # this is the old verbose >= 0
        self._print(WARNING, msg, wrap, indent)

    def error(self, msg, wrap=True, indent=""):
        """
        Print a message with verbosity level ERROR.

        Parameters
        ----------
            msg : str
                The message.

            wrap : bool, optional
                Wrap the message at 99 characters (if too long).

            indent : str, optional
                Indent each line with the this string.
        """
        # to print errors, probably not used atm
        self._print(ERROR, msg, wrap, indent)

    def critical(self, msg, wrap=True, indent=""):
        """
        Print a message with verbosity level CRITICAL.

        Parameters
        ----------
            msg : str
                The message.

            wrap : bool, optional
                Wrap the message at 99 characters (if too long).

            indent : str, optional
                Indent each line with the this string.
        """
        # just for consistency with the logging module
        self._print(CRITICAL, msg, wrap, indent)


output = Output()

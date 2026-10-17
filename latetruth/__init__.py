"""Late-arriving truth for fleet-level soft sensing."""
__version__ = "1.0.0"

import sys as _sys


def _excepthook(kind, value, traceback, default=_sys.excepthook):
    """Report a missing data archive in one line instead of a traceback."""
    from latetruth.data import MissingArchive
    if issubclass(kind, MissingArchive):
        print(f'FAIL: {value}', file=_sys.stderr, flush=True)
    else:
        default(kind, value, traceback)


_sys.excepthook = _excepthook

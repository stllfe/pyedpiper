import sys

import tqdm

import logging


class TQDMHandler(logging.StreamHandler):
    """Handler for _logging module that logs without interfering any progress bars.

    Utilizes `tqdm.write()` + sys.stdout combo for that.
    """

    def __init__(self, stream=sys.stdout):
        super().__init__(stream)

    def emit(self, record):
        try:
            message = self.format(record)
            tqdm.tqdm.write(message, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

import logging
import sys

import tqdm


class TqdmHandler(logging.StreamHandler):
    """
    Class that makes logging module to print on top of `tqdm`
    without interfering the progress bar. Basically uses `tqdm.write()` for that.
    """

    def __init__(self, stream=sys.stdout):
        super(TqdmHandler, self).__init__(stream)

    def emit(self, record):
        try:
            message = self.format(record)
            tqdm.tqdm.write(message, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

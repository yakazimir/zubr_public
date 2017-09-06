

import logging

class UtilityLogger:

    @property
    def logger(self):
        level = '.'.join([__name__,type(self).__name__])
        return logging.getLogger(level)

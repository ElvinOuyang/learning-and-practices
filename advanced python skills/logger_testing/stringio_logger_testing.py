import logging
from io import StringIO

# create logger with name 'logger_1'
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
# create two file handlers that logs debug and error level
# messages to two string buffers (which acts like two streams)
io_1 = StringIO()
fh_1 = logging.StreamHandler(io_1)
fh_1.setLevel(logging.DEBUG)

io_2 = StringIO()
fh_2 = logging.StreamHandler(io_2)
fh_2.setLevel(logging.ERROR)

# create message formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to the file handlers
fh_1.setFormatter(formatter)
fh_2.setFormatter(formatter)

# add handler to the logger
logger.addHandler(fh_1)
logger.addHandler(fh_2)


# create some functions for the testing
class Auxiliary:
    def __init__(self):
        self.logger = logging.getLogger('spam_application.auxiliary.Auxiliary')
        self.logger.info('creating an instance of Auxiliary')

    def do_something(self):
        self.logger.info('do something')
        a = 1 + 1
        self.logger.info('done doing something')
        return a


# test the loggers
logger.info('creating an instance of Auxiliary')
a = Auxiliary()
logger.info('created an instance of Auxiliary')
logger.info('calling Auxiliary.do_something()')
a.do_something()
logger.info('finished Auxiliary.do_something()')

# examine the logged strings in the string buffers
io_1.getvalue()
io_2.getvalue()

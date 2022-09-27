import logging


def get_logger() -> logging.RootLogger:
    """
    Return a generic logger to be called by the main process training a given model.
    """
    logger_name = "Train Logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    format = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(format))

    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger

import logging


def get_logger(logger_name):

    logging.basicConfig(
        level=getattr(logging, "DEBUG"),
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.pyplot").disabled = True
    logging.getLogger("matplotlib").disabled = True
    logging.getLogger("matplotlib.colorbar").disabled = True
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger = logging.getLogger(logger_name)

    return logger

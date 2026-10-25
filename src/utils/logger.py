import logging
import sys


def setup_logger(name: str = "lixai", level: str = "INFO") -> logging.Logger:
    """Proje genelinde tutarlı, Windows konsolu ve UTF-8 güvenli log çıktısı sağlayan yardımcı."""
    # Windows konsolunda UTF-8 desteğini güvenceye al
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

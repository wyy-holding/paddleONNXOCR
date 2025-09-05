import sys
from loguru import logger


class AgentLogger:
    def __init__(self):
        self.logger = logger
        self.form_str = (
            "<blue><b>{time:YYYY-MM-DD HH:mm:ss.SSS}</b></blue> "
            "<red><b>Process:{process.id}</b></red> | "
            "<green><b>{level}</b></green> | "
            "<red><b>{file}</b></red> | <red><b>{module}</b></red> | "
            "<red><b>{function}:{line}</b></red> ----> {message}"
        )
        self.logger.remove()
        self.logger.add(sys.stdout, format=self.form_str, enqueue=True)


log = AgentLogger().logger

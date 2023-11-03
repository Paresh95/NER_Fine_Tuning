import logging
from abc import ABC, abstractmethod


class BaseModelTraining(ABC):
    def __init__(self, logging_file_path):
        self.logging_file_path = logging_file_path
        logging.basicConfig(
            filename=self.logging_file_path,
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging

    @abstractmethod
    def training_logic(self) -> None:
        raise NotImplementedError

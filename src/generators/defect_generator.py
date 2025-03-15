from abc import abstractmethod, ABC
import numpy as np


class DefectGenerator(ABC):
    @abstractmethod
    def generate_defects(self) -> np.ndarray:
        """Генерирует дефект на изображении."""
        pass

    @abstractmethod
    def highlight_defects(self) -> np.ndarray:
        """Генерирует дефект на изображении."""
        pass

from abc import ABC, abstractmethod
import numpy as np


class SurfaceAbs(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform_to_camera(self, view_matrix: np.ndarray):
        """This method must be overridden in subclasses"""
        pass

from abc import ABC, abstractmethod
import numpy as np
from Ray import Ray

class SurfaceAbs(ABC):
    @abstractmethod
    def transform_to_camera(self, view_matrix: np.ndarray):
        """This method must be overridden in subclasses"""
        pass

    @abstractmethod
    def intersect(self, ray: Ray):
        """ This method must be overridden in subclasses"""
        pass



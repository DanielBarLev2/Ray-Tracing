from abc import ABC, abstractmethod
import numpy as np
from Ray import Ray


class SurfaceAbs(ABC):
    def __init__(self, material_index):
        self.material_index = material_index

    @abstractmethod
    def transform_to_camera(self, view_matrix: np.ndarray):
        """This method must be overridden in subclasses"""
        pass

    @abstractmethod
    def intersect(self, rays_source: np.ndarray, ray_vectors: np.ndarray) -> np.ndarray:
        """ This method must be overridden in subclasses"""
        pass



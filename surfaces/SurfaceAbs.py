from abc import ABC, abstractmethod
from util import *


class SurfaceAbs(ABC):
    def __init__(self, material_index, index):
        self.index = index
        self.material_index = material_index

    def __repr__(self):
        return f"{self.__class__.__name__}(material_index={self.material_index})"

    @abstractmethod
    def transform_to_camera(self, view_matrix: np.ndarray):
        """This method must be overridden in subclasses"""
        pass

    @abstractmethod
    def intersect(self, ray_source: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
        """ This method must be overridden in subclasses"""
        pass

    @abstractmethod
    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray) -> np.ndarray:
        """ This method must be overridden in subclasses"""
        pass

    @abstractmethod
    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        """ This method must be overridden in subclasses"""
        pass

    def get_material_index(self):
        return self.material_index - 1  # original indices start from 1

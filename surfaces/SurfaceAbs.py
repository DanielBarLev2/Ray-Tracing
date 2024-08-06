from util import *
from abc import ABC, abstractmethod


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

    @abstractmethod
    def calculate_normals(self, rays_interactions: np.ndarray) -> np.ndarray:
        """ This method must be overridden in subclasses"""
        pass

    def get_material_index(self):
        if self.material_index:
            return self.material_index
        else:
            return 0


def get_surfaces_normals(surfaces: list, surfaces_indices: np.ndarray, ray_hits: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(ray_hits)

    for idx in np.unique(surfaces_indices):
        if idx < 0 or idx >= len(surfaces):
            continue
        surface: SurfaceAbs = surfaces[idx]
        mask = (surfaces_indices == idx)
        rays_interactions_with_surface = ray_hits[mask]
        normals[mask] = surface.calculate_normals(rays_interactions_with_surface)

    return normals


def get_surfaces_material_indices(surfaces: list, surfaces_indices: np.ndarray) -> np.ndarray:
    material_indices = np.zeros_like(surfaces_indices)

    for idx in np.unique(surfaces_indices):
        if idx < 0 or idx >= len(surfaces):
            continue
        surface: SurfaceAbs = surfaces[idx]
        mask = (surfaces_indices == idx)
        material_indices[mask] = surface.get_material_index()

    return material_indices
